import cv2
import numpy as np 
import glob 
import os
import trimesh
import ffmpeg
import gc
import argparse
import pyrender
import pickle
import pymeshlab as pmlab
import matplotlib


def render_single_mesh_sequence(seq_path):
    tmp_dir = 'renders/tmp'
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
 
    # img_filename = '/local/home/yaqqin/datasets/multiface/mini_dataset/m--20180227--0000--6795937--GHS/images/E057_Cheeks_Puffed/400002/021897.png'
    # img = cv2.imread(img_filename)
    # h, w, _ = img.shape
    # print(f"image shape is {img.shape}")
    
    h, w = 640, 480

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    
    material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor= (144, 117, 87, 255),
                metallicFactor=0.8,
                roughnessFactor=0.8,
                wireframe=True,
                emissiveFactor= 0.3,
    )

    r = pyrender.OffscreenRenderer(h, w)

    print(seq_path)

    video_woA_path = os.path.join(tmp_dir, "render.mp4")
    video = cv2.VideoWriter(video_woA_path, fourcc, fps, (h, w))

    obj_paths = glob.glob(f'{seq_path}/*.obj')
    obj_paths.sort(key = lambda x : os.path.split(x)[-1][:-4])
    
    for i, obj_path in enumerate(obj_paths):
        
        obj_mesh = trimesh.load_mesh(obj_path, process=False)
        
        py_mesh = pyrender.Mesh.from_trimesh(obj_mesh, material=material)

        scene = pyrender.Scene(bg_color=[0, 0, 0, 255],
                               ambient_light=[0.2, 0.2, 0.2])
        node = pyrender.Node(
            mesh=py_mesh,
            translation=[0, 0, 0]
        )
        scene.add_node(node)

        base, frame_name = os.path.split(obj_path)
        camera_pose = np.loadtxt(os.path.join(base, f'{frame_name[:-4]}_transform.txt')) # 3x4
        camera_pose = np.concatenate((camera_pose, [[0, 0, 0, 1]]), axis=0)   # to 4x4
        camera_pose[:, 3] += [0, 0, -300, 0]   # translate the cam pose to catch the obj
        scene.add(cam, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        color, _ = r.render(scene)        
            
        output_frame = os.path.join(tmp_dir, f"frame_{i}_{frame_name[:-4]}.png")

        cv2.imwrite(output_frame, color)
        frame = cv2.imread(output_frame)
        if i == 0:
            print(f"shape of color = {color.shape}")
            print(f"shape of frame = {frame.shape}")
        video.write(frame)

    video.release()
    del obj_mesh
    gc.collect()

def render_one_mesh(obj_path):
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    
    base, frame_name = os.path.split(obj_path)
    camera_pose = np.loadtxt(os.path.join(base, f'{frame_name[:-4]}_transform.txt')) # 3x4
    camera_pose = np.concatenate((camera_pose, [[0, 0, 0, 1]]), axis=0)   # to 4x4

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0) # by default white light
    
    material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor= (144, 117, 87, 255),
                metallicFactor=0.8,
                roughnessFactor=0.8,
                wireframe=True,
                emissiveFactor= 0.3,
    )
    h, w = 2048, 1334
    r = pyrender.OffscreenRenderer(h, w)
    
    obj_mesh = trimesh.load_mesh(obj_path, process=False)
    py_mesh = pyrender.Mesh.from_trimesh(obj_mesh, material=material)

    scene = pyrender.Scene(bg_color=[0, 0, 0, 255],                  # by default black
                            ambient_light=[.2, .2, .2])
    node = pyrender.Node(
        mesh=py_mesh,
        translation=[0, 0, 0]
    )
    scene.add_node(node)
    cam_pos = camera_pose
    cam_pos[:, 3] += [0, 0, -500, 0]
    scene.add(cam, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    pyrender.Viewer(scene)



def images_to_video(folder_path):
    img_arr = []
    for filename in glob.glob(f'{folder_path}/*.png'):
        img = cv2.imread(filename)
        height, width, _ = img.shape
        size = (width, height)
        img_arr.append(img)
    print(f'there are {len(img_arr)} images under this folder')
    print(f'img size: h={size[1]}, w={size[0]}')
    
    out = cv2.VideoWriter(f'{folder_path}/video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(img_arr)):
        out.write(img_arr[i])
    out.release()
    
    

def main():
    entity_path = 'm--20180227--0000--6795937--GHS'
    assets = 'tracked_mesh'
    eid = 'E057_Cheeks_Puffed'
    seq_path = os.path.join(entity_path, assets, eid)
    render_single_mesh_sequence(seq_path)
    
    
    # obj_path = 'm--20180227--0000--6795937--GHS/tracked_mesh/E057_Cheeks_Puffed/021897.obj'
    # render_one_mesh(obj_path)
        
   
if __name__ == '__main__':
    main()