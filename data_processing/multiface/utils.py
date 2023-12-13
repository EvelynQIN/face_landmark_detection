# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nvdiffrast batched rendering
import cv2
import numpy as np
# import nvdiffrast.torch as dr
import torch
from PIL import Image

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# obj file dataset
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from numpy.lib.format import MAGIC_PREFIX
from PIL import Image
from torch.autograd import Variable


def load_obj(filename):
    vertices = []
    faces_vertex, faces_uv = [], []
    uvs = []
    with open(filename, "r") as f:
        for s in f:
            l = s.strip()
            if len(l) == 0:
                continue
            parts = l.split(" ")
            if parts[0] == "vt":
                uvs.append([float(x) for x in parts[1:]])
            elif parts[0] == "v":
                vertices.append([float(x) for x in parts[1:]])
            elif parts[0] == "f":
                faces_vertex.append([int(x.split("/")[0]) for x in parts[1:]])
                faces_uv.append([int(x.split("/")[1]) for x in parts[1:]])
    # make sure triangle ids are 0 indexed
    obj = {
        "verts": np.array(vertices, dtype=np.float32),
        "uvs": np.array(uvs, dtype=np.float32),
        "vert_ids": np.array(faces_vertex, dtype=np.int32) - 1,
        "uv_ids": np.array(faces_uv, dtype=np.int32) - 1,
    }
    return obj


def check_path(path):
    if not os.path.exists(path):
        sys.stderr.write("%s does not exist!\n" % (path))
        sys.exit(-1)


def load_krt(path):
    cameras = {}

    with open(path, "r") as f:
        while True:
            name = f.readline()
            if name == "":
                break

            intrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            dist = [float(x) for x in f.readline().split()]
            extrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            f.readline()

            cameras[name[:-1]] = {
                "intrin": np.array(intrin),
                "dist": np.array(dist),
                "extrin": np.array(extrin),
            }

    return cameras


def compute_M(cam_id, expression_id, frame_id):
    
    # get camera calibrations given the cam_id
    krt_dir = '../../datasets/multiface/minidataset/m--20180227--0000--6795937--GHS/KRT'
    krts = load_krt(krt_dir)
    krt = krts[cam_id]
    
    # compute view directions of the camera
    extrin = krt["extrin"]  # cam [R|T]
    campos = -np.dot(extrin[:3, :3].T, extrin[:3, 3])   # (3, )
    # print("shape of campos: ", campos.shape)    
    
    # geometry
    mesh_path = '../../datasets/multiface/minidataset/m--20180227--0000--6795937--GHS/tracked_mesh'
    
    # view direction of this frame (head pose of the mesh)
    transf = np.genfromtxt(
        "{}/{}/{}_transform.txt".format(mesh_path, expression_id, frame_id)
    )
    R_f = transf[:3, :3]
    t_f = transf[:3, 3]
    campos = np.dot(R_f.T, campos - t_f).astype(np.float32)
    view = campos / np.linalg.norm(campos)

    extrin, intrin = krt["extrin"], krt["intrin"]
    R_C = extrin[:3, :3]
    t_C = extrin[:3, 3]
    camrot = np.dot(R_C, R_f).astype(np.float32)
    camt = np.dot(R_C, t_f) + t_C
    camt = camt.astype(np.float32)

    M = intrin @ np.hstack((camrot, camt[None].T))  # (3, 4)
    # print("shape of M: ", M.shape)
    return M


def project_3dmesh_to_view(cam_id, expression_id, frame_id, resolution=[2048, 1334]):
    
    # mesh
    mesh_path = '../../datasets/multiface/minidataset/m--20180227--0000--6795937--GHS/tracked_mesh'
    path = "{}/{}/{}.bin".format(mesh_path, expression_id, frame_id)
    verts = np.fromfile(path, dtype=np.float32) # (7306*3, )
    verts = torch.from_numpy(verts).reshape(1, -1, 3)   # (bs, v, 3)
    
    # proj matrix computation
    M = compute_M(cam_id, expression_id, frame_id)
    M = torch.from_numpy(M).unsqueeze(0).float()
    print("shape of M: ", M.shape)

    ones = torch.ones((verts.shape[0], verts.shape[1], 1))  # (bs, v, 1)
    pos_homo = torch.cat((verts, ones), -1) # (bs, v, 4)
    projected = torch.bmm(M, pos_homo.permute(0, 2, 1)) 
    projected = projected.permute(0, 2, 1) 
    
    # rescale of 3d proj points w.r.t. resolution -> to the NDC (-1, 1)
    proj = torch.zeros_like(projected)
    proj[..., 0] = (
        projected[..., 0] / (resolution[1] / 2) - projected[..., 2]
    ) / projected[..., 2]
    proj[..., 1] = (
        projected[..., 1] / (resolution[0] / 2) - projected[..., 2]
    ) / projected[..., 2]
    clip_space, _ = torch.max(projected[..., 2], 1, keepdim=True)
    proj[..., 2] = projected[..., 2] / clip_space   # (bs, v, 3) homo positions
    print("shape of proj", proj.shape)

    return proj   


def NDC_to_raster(points, width, height):
    """Normalize the projected points to fit within the raster coordinates of the 2D image."""
    
    # Initialize the normalized points list
    normalized_points = []
    
    # Normalize each point
    for point in points:
        normalized_x = (point[0] + 1) * (width / 2) + point[0]
        normalized_y = (point[1] + 1) * (height / 2) + point[1]
        normalized_points.append((int(normalized_x), int(normalized_y)))
    
    return normalized_points


def show_proj_result(cam_id, expression_id, frame_id, resolution=[2048, 1334], mask = None):
    h, w = resolution
    proj = project_3dmesh_to_view(cam_id, expression_id, frame_id, resolution=resolution).numpy().reshape(-1, 3)
    print(np.max(proj, axis = 0))
    print(np.min(proj, axis = 0))
    
    proj_ras = NDC_to_raster(proj, w, h)
    print(np.max(proj_ras, axis = 0))
    print(np.min(proj_ras, axis = 0))
    
    if mask:
        proj = proj[mask]
    # image
    img_path = "../../datasets/multiface/minidataset/m--20180227--0000--6795937--GHS/images"
    path = "{}/{}/{}/{}.png".format(img_path, expression_id, cam_id, frame_id)
    image = cv2.imread(path) 
    print(image.shape)
    
    for px, py in proj_ras:
        if px > w or py > h or px < 0 or py < 0:
            continue
        cv2.circle(image, (px, py), 1, (255, 255, 255), -1) 
        
    cv2.imwrite("test_projection.png", image)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

    
    
    
    