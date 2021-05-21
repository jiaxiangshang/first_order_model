import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch

# self
_curr_path = os.path.abspath(__file__)  # /home/..../face
_cur_dir = os.path.dirname(_curr_path)  # ./
_tf_dir = os.path.dirname(_cur_dir)  # ./
print(_tf_dir)
sys.path.append(_tf_dir)  # /home/..../pytorch3d

_dl_dir = os.path.dirname(_tf_dir)  # ./
_deep_learning_dir = os.path.dirname(_dl_dir)  # ../
print(_deep_learning_dir)
sys.path.append(_deep_learning_dir)  # /home/..../pytorch3d

from first_order_model.sync_batchnorm import DataParallelWithCallback
from first_order_model.modules.generator import OcclusionAwareGenerator
from first_order_model.modules.keypoint_detector import KPDetector
from first_order_model.animate import normalize_kp
from scipy.spatial import ConvexHull



# save result
from base.io import *


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num

"""
python demo.py --config config/vox-256.yaml \
--dic_dataset /media/jiaxiangshang/My\ Passport/0_SHANG_DATA/1_Face_2D/7_voxel_celeb2_val_GL_unique --name_global_list train_video_10 \
--dic_save /media/jiaxiangshang/My\ Passport/1_SHANG_EXP/2_frrnet \
--checkpoint /data0/2_Project/python/deeplearning_python/dl_model_reen/vox-cpk.pth.tar \
--relative --adapt_scale


python ./first_order_model/0_test_gl_img2img.py \
--config config/vox-256.yaml \
--dic_dataset /apdcephfs/private_alexinwang/jxshang/data/0_3DFace_Train/2_mono/7_voxel_celeb2_val_GL_unique_5 \
--name_global_list train_video_5 \
--dic_save /apdcephfs/share_782420/jxshang/exp/5_reen_results/first_order_model \
--checkpoint /apdcephfs/private_alexinwang/jxshang/project/deeplearning_python/dl_model_reen/vox-cpk.pth.tar \
--relative \
--adapt_scale

python ./first_order_model/0_test_gl_img2img.py \
--config config/vox-256.yaml \
--dic_dataset /apdcephfs/private_alexinwang/jxshang/data/0_3DFace_Train/2_mono/7_voxel_celeb2_val_GL_unique_5 \
--name_global_list train_video_5 \
--dic_save /apdcephfs/share_782420/jxshang/exp/6_reen_quati/first_order_model \
--checkpoint /apdcephfs/private_alexinwang/jxshang/project/deeplearning_python/dl_model_reen/vox-cpk.pth.tar \
--relative \
--adapt_scale \
--flag_quati 1

"""
from first_order_model.crop_video import *
def test_video(opt, path_src, list_path_tar):
    path_src_pure, _ = os.path.splitext(path_src)
    path_src_bbox = path_src_pure + '_bbox.txt'
    src_bbox = parse_self_facebbox(path_src_bbox)[:-1]

    source_image_ori = imageio.imread(path_src)
    source_image, _, bbox_src = crop_bbox(source_image_ori, src_bbox)


    driving_video_ori = []
    driving_video = []
    list_m_inv = []
    list_bbox = []
    for i in range(len(list_path_tar)):
        path_tar = list_path_tar[i]

        path_tar_pure, _ = os.path.splitext(path_tar)
        path_tar_bbox = path_tar_pure + '_bbox.txt'
        tar_bbox = parse_self_facebbox(path_tar_bbox)[:-1]

        tar_image_ori = imageio.imread(path_tar)
        tar_image, m_inv, bbox = crop_bbox(tar_image_ori, tar_bbox)

        driving_video_ori.append(tar_image_ori)
        driving_video.append(tar_image)
        list_m_inv.append(m_inv)
        list_bbox.append(bbox)

    #source_image_ori = resize(source_image_ori, (256, 256))[..., :3]
    source_image = resize(source_image, (256, 256))[..., :3]
    #driving_video_ori = [resize(frame, (256, 256))[..., :3] for frame in driving_video_ori]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    opt.config = os.path.join(_cur_dir, opt.config)
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)

    if opt.find_best_frame or opt.best_frame is not None:
        i = opt.best_frame if opt.best_frame is not None else find_best_frame(source_image, driving_video, cpu=opt.cpu)
        print("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i + 1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector,
                                             relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
        predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector,
                                              relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=opt.relative,
                                     adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
    list_result = [img_as_ubyte(frame) for frame in predictions]
    return source_image_ori, driving_video_ori, list_result, list_m_inv, bbox_src, list_bbox
    #imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)

import ast
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='config/vox-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='/data0/2_Project/python/deeplearning_python/dl_model_reen/vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='sup-mat/source.png', help="path to driving video")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")
 
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,  
                        help="Set frame to start from.")
 
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    # jiaxiang
    parser.add_argument('--dic_dataset', default='/media/jiaxiangshang/My Passport/0_SHANG_DATA/1_Face_2D/7_voxel_celeb2_val_GL_unique', type=str, help='')
    parser.add_argument('--dic_save', default='/media/jiaxiangshang/My Passport/1_SHANG_EXP/2_frrnet/1_free_vc', type=str, help='')

    parser.add_argument('--name_global_list', default='train_video_10', type=str, help='')

    parser.add_argument('--num_src_k', default=1, type=int, help='')
    parser.add_argument('--num_tar_k', default=10, type=int, help='')

    parser.add_argument('--flag_quati', default=0, type=ast.literal_eval, help='')

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    # read global list
    emotion_list, dic_folderLeaf_list, dict_video_2_frames = parse_video_global_list(opt.dic_dataset, opt.name_global_list, True)

    # save global list
    if os.path.isdir(opt.dic_save) == False:
        os.makedirs(opt.dic_save)

    path_train_list = os.path.join(opt.dic_save, "eval.txt")
    f_train_global = open(path_train_list, 'w')

    list_name_videoKey = list(dict_video_2_frames.keys())
    for i in range(len(list_name_videoKey)):
        print('Sample', i)
        name_vk = list_name_videoKey[i]
        list_frames = dict_video_2_frames[name_vk]

        step = int(len(list_frames)/opt.num_src_k)
        for j in range(0, len(list_frames), step):
            main_frame = list_frames[j]

            for i_v in range(len(list_name_videoKey)):
                if opt.flag_quati:
                    if i_v != i:
                        continue
                else:
                    if i_v % opt.num_tar_k != 0 and i_v != i:
                        continue
                name_vk_SEAR = list_name_videoKey[i_v]
                list_frames_SEAR = dict_video_2_frames[name_vk_SEAR]
                list_path_SEAR = [lf+'.jpg' for lf in list_frames_SEAR]
                source_image, list_driving_video, list_result, list_m_inv, bbox_src, list_bbox_tar = test_video(opt, main_frame + '.jpg', list_path_SEAR)

                name_subfolder_save_0 = 'reen_%d' % (i)
                name_subfolder_save = 'numf_%d_on_%d' % (j, i_v)
                dic_subf_save = os.path.join(opt.dic_save, name_subfolder_save_0+'/'+name_subfolder_save)
                print('save subdic', dic_subf_save)
                if os.path.isdir(dic_subf_save) == False:
                    os.makedirs(dic_subf_save)

                for f in range(len(list_frames_SEAR)):
                    path_frame_pure = list_frames_SEAR[f]
                    _, name_frame = os.path.split(path_frame_pure)

                    path_save_src = os.path.join(dic_subf_save, name_frame + '_src.jpg')
                    path_save = os.path.join(dic_subf_save, name_frame + '.jpg')
                    path_all_save = os.path.join(dic_subf_save, name_frame + '_concat.jpg')

                    src_img = source_image
                    tar_img = list_driving_video[f]
                    result_img = list_result[f]
                    M_inv = list_m_inv[f]
                    bbox_tar = list_bbox_tar[f]
                    if 1:
                        from base.io import inverse_affine_warp_overlay
                        result_img_replace = inverse_affine_warp_overlay(M_inv, tar_img, result_img * 1.0, np.ones_like(result_img), flag_cv=True)
                        # # visual
                        # cv2.imshow("Image Debug", result_img)
                        # k = cv2.waitKey(0) & 0xFF
                        # if k == 27:
                        #     cv2.destroyAllWindows()
                        # cv2.imshow("Image Debug", cv2.cvtColor(img_replace, cv2.COLOR_RGB2BGR))
                        # k = cv2.waitKey(0) & 0xFF
                        # if k == 27:
                        #     cv2.destroyAllWindows()
                    result_concat = np.concatenate([src_img, tar_img, result_img_replace], axis=1)
                    result_concat = result_concat.astype(np.uint8)
                    # save
                    src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
                    result_img_replace = cv2.cvtColor(result_img_replace, cv2.COLOR_RGB2BGR)
                    result_concat = cv2.cvtColor(result_concat, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(path_save_src, src_img)
                    cv2.imwrite(path_save, result_img_replace)
                    cv2.imwrite(path_all_save, result_concat)

                    path_save_bbox = os.path.join(dic_subf_save, name_frame + '_bbox_fom_src.txt')
                    write_self_facebbox(path_save_bbox, bbox_src)
                    path_save_bbox = os.path.join(dic_subf_save, name_frame + '_bbox_fom.txt')
                    write_self_facebbox(path_save_bbox, bbox_tar)

                    f_train_global.write("%s %s\n" % (name_subfolder_save_0 + '/' + name_subfolder_save, name_frame))

