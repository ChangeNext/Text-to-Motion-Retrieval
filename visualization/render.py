import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os
os.environ['PYOPENGL_PLATFORM'] = "osmesa"

import torch
from visualize.simplify_loc2rot import joints2smpl
import pyrender
import matplotlib.pyplot as plt

import io
import imageio
from shapely import geometry
import trimesh
from pyrender.constants import RenderFlags
from glob import glob
from PIL import Image

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P

def render_gif(motions, name, outdir='render_vis', device_id=0):
    frames, njoints, nfeats = motions.shape
    MINS = motions.min(axis=0).min(axis=0)
    MAXS = motions.max(axis=0).max(axis=0)

    height_offset = MINS[1]
    motions[:, :, 1] -= height_offset
    trajec = motions[:, 0, [0, 2]]

    j2s = joints2smpl(num_frames=frames, device_id=0, cuda=True)
    rot2xyz = Rotation2xyz(device=torch.device("cuda:0"))
    faces = rot2xyz.smpl_model.faces

    if not os.path.exists(os.path.join(outdir, name+'.pt')):
        print(f'Running SMPLify, it may take a few minutes.')
        motion_tensor, opt_dict = j2s.joint2smpl(motions)  # [nframes, njoints, 3]

        vertices = rot2xyz(torch.tensor(motion_tensor).clone(), mask=None,
                                        pose_rep='rot6d', translation=True, glob=True,
                                        jointstype='vertices',
                                        vertstrans=True)

        torch.save(vertices, os.path.join(outdir, name+'.pt'))
    else:
        vertices = torch.load(os.path.join(outdir, name+'.pt'))

    frames = vertices.shape[3] # shape: 1, nb_frames, 3, nb_joints
    print (vertices.shape)
    MINS = torch.min(torch.min(vertices[0], axis=0)[0], axis=1)[0]
    MAXS = torch.max(torch.max(vertices[0], axis=0)[0], axis=1)[0]
    # vertices[:,:,1,:] -= MINS[1] + 1e-5

    out_list = []

    minx = MINS[0] - 0.5
    maxx = MAXS[0] + 0.5
    minz = MINS[2] - 0.5 
    maxz = MAXS[2] + 0.5
    polygon = geometry.Polygon([[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]])
    polygon_mesh = trimesh.creation.extrude_polygon(polygon, 1e-5)

    vid = []
    for i in range(frames):
        mesh = Trimesh(vertices=vertices[0, :, :, i].squeeze().tolist(), faces=faces)

        # base_color = (0.11, 0.53, 0.8, 0.5)
        min_alpha = 0.5
        alpha_value = min_alpha + (i / frames) * (1 - min_alpha)
        base_color = (0.1, 0.2, 0.6, 0.5) 
        ## OPAQUE rendering without alpha
        ## BLEND rendering consider alpha 
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7,
            alphaMode='OPAQUE',
            baseColorFactor=base_color
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        # polygon_mesh.visual.face_colors = [0, 0, 0, 0.21]
        # polygon_mesh.visual.face_colors = [200, 200, 200, 100]
        polygon_mesh.visual.face_colors = [70, 70, 70, 1.0]
        polygon_render = pyrender.Mesh.from_trimesh(polygon_mesh, smooth=False)

        bg_color = [1, 1, 1, 0.8]
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))

        sx, sy, tx, ty = [0.75, 0.75, 0, 0.10]

        camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))

        light = pyrender.DirectionalLight(color=[1,1,1], intensity=300)

        scene.add(mesh)

        c = np.pi / 2

        scene.add(polygon_render, pose=np.array([[ 1, 0, 0, 0],

        [ 0, np.cos(c), -np.sin(c), MINS[1].cpu().numpy()],

        [ 0, np.sin(c), np.cos(c), 0],

        [ 0, 0, 0, 1]]))

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = [0, 1, 1]
        scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = [1, 1, 2]
        scene.add(light, pose=light_pose.copy())


        c = -np.pi / 6

        scene.add(camera, pose=[[ 1, 0, 0, (minx+maxx).cpu().numpy()/2],

                                [ 0, np.cos(c), -np.sin(c), 1.5],

                                [ 0, np.sin(c), np.cos(c), max(4, minz.cpu().numpy()+(1.5-MINS[1].cpu().numpy())*2, (maxx-minx).cpu().numpy())+0.5],

                                [ 0, 0, 0, 1]
                                ])

        # render scene
        r = pyrender.OffscreenRenderer(960, 960)

        color, _ = r.render(scene, flags=RenderFlags.RGBA)
        # Image.fromarray(color).save(outdir+name+'_'+str(i)+'.png')

        vid.append(color)

        r.delete()

    out = np.stack(vid, axis=0)
    imageio.mimsave(os.path.join(outdir, f'{name}_render.gif'), out, fps=20)

'''
def render_image(motions, name, outdir='test_vis', device_id=0):
    frames, njoints, nfeats = motions.shape
    MINS = motions.min(axis=0).min(axis=0)
    MAXS = motions.max(axis=0).max(axis=0)

    height_offset = MINS[1]
    motions[:, :, 1] -= height_offset
    trajec = motions[:, 0, [0, 2]]

    j2s = joints2smpl(num_frames=frames, device_id=0, cuda=True)
    rot2xyz = Rotation2xyz(device=torch.device("cuda:0"))
    faces = rot2xyz.smpl_model.faces

    if not os.path.exists(os.path.join(outdir, name+'.pt')):
        print(f'Running SMPLify, it may take a few minutes.')
        motion_tensor, opt_dict = j2s.joint2smpl(motions)  # [nframes, njoints, 3]

        vertices = rot2xyz(torch.tensor(motion_tensor).clone(), mask=None,
                                        pose_rep='rot6d', translation=True, glob=True,
                                        jointstype='vertices',
                                        vertstrans=True)

        # torch.save(vertices, os.path.join(outdir, name+'.pt'))
    else:
        vertices = torch.load(os.path.join(outdir, name+'.pt'))
        
    frames = vertices.shape[3] # shape: 1, nb_frames, 3, nb_joints
    print (vertices.shape)
    MINS = torch.min(torch.min(vertices[0], axis=0)[0], axis=1)[0]
    MAXS = torch.max(torch.max(vertices[0], axis=0)[0], axis=1)[0]
    # vertices[:,:,1,:] -= MINS[1] + 1e-5


    out_list = []
    
    minx = MINS[0] - 0.5
    maxx = MAXS[0] + 0.5
    minz = MINS[2] - 0.5 
    maxz = MAXS[2] + 0.5
    polygon = geometry.Polygon([[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]])
    polygon_mesh = trimesh.creation.extrude_polygon(polygon, 1e-5)

    vid = []

    bg_color = [1, 1, 1, 0.8]
    scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))
    for i in range(frames):
        if i % 15 != 0:
            continue

        mesh = Trimesh(vertices=vertices[0, :, :, i].squeeze().tolist(), faces=faces)

        base_color = (0.11, 0.53, 0.8, 0.5)
        ## OPAQUE rendering without alpha
        ## BLEND rendering consider alpha 
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7,
            alphaMode='OPAQUE',
            baseColorFactor=base_color, roughnessFactor = 0.7
        )


        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        polygon_mesh.visual.face_colors = [0, 0, 0, 0.21]
        polygon_render = pyrender.Mesh.from_trimesh(polygon_mesh, smooth=False)

        
        sx, sy, tx, ty = [0.75, 0.75, 0, 0.10]

        camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))

        light = pyrender.DirectionalLight(color=[1,1,1], intensity=300)

        scene.add(mesh)

    c = np.pi / 2

    scene.add(polygon_render, pose=np.array([[ 1, 0, 0, 0],

    [ 0, np.cos(c), -np.sin(c), MINS[1].cpu().numpy()],

    [ 0, np.sin(c), np.cos(c), 0],

    [ 0, 0, 0, 1]]))

    light_pose = np.eye(4)
    light_pose[:3, 3] = [0, -1, 1]
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = [0, 1, 1]
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = [1, 1, 2]
    scene.add(light, pose=light_pose.copy())


    c = -np.pi / 6

    scene.add(camera, pose=[[ 1, 0, 0, (minx+maxx).cpu().numpy()/2],

                            [ 0, np.cos(c), -np.sin(c), 1.5],

                            [ 0, np.sin(c), np.cos(c), 0.5+max(4, minz.cpu().numpy()+(1.5-MINS[1].cpu().numpy())*2, (maxx-minx).cpu().numpy())+2],

                            [ 0, 0, 0, 1]
                            ])
    
    # render scene
    r = pyrender.OffscreenRenderer(960, 960)

    color, _ = r.render(scene, flags=RenderFlags.RGBA)
    Image.fromarray(color).save(os.path.join(outdir, f'{name}_render.png'))
'''
def render_image(motions, name, outdir='test_vis', device_id=0):
    frames, njoints, nfeats = motions.shape
    MINS = motions.min(axis=0).min(axis=0)
    MAXS = motions.max(axis=0).max(axis=0)

    height_offset = MINS[1]
    motions[:, :, 1] -= height_offset
    trajec = motions[:, 0, [0, 2]]

    j2s = joints2smpl(num_frames=frames, device_id=device_id, cuda=True)
    rot2xyz = Rotation2xyz(device=torch.device(f"cuda:{device_id}"))
    faces = rot2xyz.smpl_model.faces

    if not os.path.exists(os.path.join(outdir, name + '.pt')):
        print(f'Running SMPLify, it may take a few minutes.')
        motion_tensor, opt_dict = j2s.joint2smpl(motions)  # [nframes, njoints, 3]

        vertices = rot2xyz(torch.tensor(motion_tensor).clone(), mask=None,
                           pose_rep='rot6d', translation=True, glob=True,
                           jointstype='vertices',
                           vertstrans=True)

        torch.save(vertices, os.path.join(outdir, name + '.pt'))
    else:
        vertices = torch.load(os.path.join(outdir, name + '.pt'))

    frames = vertices.shape[3]  # shape: 1, nb_frames, 3, nb_joints
    print(vertices.shape)
    MINS = torch.min(torch.min(vertices[0], axis=0)[0], axis=1)[0]
    MAXS = torch.max(torch.max(vertices[0], axis=0)[0], axis=1)[0]

    out_list = []

    minx = MINS[0] - 0.5
    maxx = MAXS[0] + 0.5
    minz = MINS[2] - 0.5
    maxz = MAXS[2] + 0.5
    polygon = geometry.Polygon([[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]])
    polygon_mesh = trimesh.creation.extrude_polygon(polygon, 1e-5)

    bg_color = [1, 1, 1, 0.8]
    scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))

    # capture_interval = max(1, frames // 2)  
    print(frames)
    # for i in range(0, frames, capture_interval):
    
    for i in range(0, frames):
        # if i % 23 != 0:
        #     continue
        if  i==60:
            mesh = Trimesh(vertices=vertices[0, :, :, i].squeeze().tolist(), faces=faces)

            # min_alpha = 0.7  # 최소 알파 값 설정 (0.3는 예시 값으로, 원하는 값으로 변경 가능)
            # min_alpha = 0.6
            # max_alpha = 1.0
            # alpha_value = min_alpha + (i / frames) * (1 - min_alpha)
            # period = frames
            # alpha_value = min_alpha + (max_alpha - min_alpha) * (0.5 * (1 + np.sin(2 * np.pi * i / period)))
            base_color = (0.1, 0.2, 0.7, 1.0)
            
            # if i==0:
            #     # base_color = (0.8, 0.2, 0.2, 0.4)
            #     base_color = (0.1, 0.2, 0.7, 0.4)
            #     # base_color = (0.1, 0.2, 0.6, 0.7)
            # if i==20:
            #     base_color = (0.1, 0.2, 0.7, 0.8)
            # if i==70:
            #     base_color = (0.1, 0.2, 0.7, 0.9)
            # if i==50:
            #     base_color = (0.1, 0.2, 0.7, 0.7)
            # if i==120:
            #     base_color = (0.1, 0.2, 0.7, 0.9)
            # if i==130:
            #     base_color = (0.1, 0.2, 0.7, 1.0)
            # if i==115:
            #     base_color = (0.1, 0.2, 0.7, 1.0)
            # # if i == 20:
            #     # base_color = (0.8, 0.2, 0.2, 0.4)
            #     # base_color = (0.1, 0.2, 0.7,1.0)
            # if i == 30:
            #     base_color = (0.1, 0.2, 0.7, 1.0)
            # if i == 50:
            #     base_color = (0.1, 0.2, 0.7, 0.6)
            # if i == 120:
            #     base_color = (0.1, 0.2, 0.7, 0.9)
                
                # base_color = (0.1, 0.2, 0.7, 0.6)
            # if i == 111:
            #     base_color = (0.1, 0.2, 0.7, 0.8)
            #     base_color = (0.3, 0.7, 0.3, 0.8)
            # if i == 70:
            #     base_color = (0.1, 0.2, 0.6, 0.7)
            # if i == 162:
            #     base_color = (1.0, 0.5, 0.5, 0.9)
            # if i == 100:
            #     base_color = (0.1, 0.2, 0.6, 0.9)
            #     # base_color = (0.1, 0.2, 0.6, 0.7)
            #     # base_color = (0.3, 0.7, 0.3, 0.5)
            # #     # base_color = (0.1, 0.2, 0.6, 0.7)
            # #     base_color = (1.0, 0.5, 0.5, 0.9)
            # if i == 92:
            #     base_color = (0.8, 0.2, 0.2, 0.8)
            #     # base_color = (0.1, 0.2, 0.6, 0.7)
            #     # base_color = (1.0, 0.5, 0.5, 0.8)
            #     # base_color = (0.1, 0.2, 0.6, 0.7)
            #     # base_color = (0.1, 0.2, 0.6, 1.0)
            # #     base_color = (0.3, 0.8, 0.3, 0.8)

            #     base_color = (0.3, 0.8, 0.3, 0.7)
            #     # base_color = (0.3, 0.7, 0.3, 0.9)
            #     # base_color = (0.3, 0.7, 0.3, 1.0)
            #     base_color = (0.3, 0.8, 0.3, 0.9)
            #     # base_color = (0.1, 0.2, 0.6, 1.0)
            #     # base_color = (1.0, 0.5, 0.5, 0.9)
            #     # base_color = (0.1, 0.2, 0.6, 0.7)
            # #     # base_color = (0.1, 0.2, 0.6, 0.7)
            # #     base_color = (1.0, 0.5, 0.5, 0.9)
            # # else:
            # #     base_color = (0.1, 0.2, 0.6, 1.0)
            # # if i == 145:
            # #     base_color = (1.0, 0.5, 0.5, 0.9)
            # # base_color = (0.1, 0.2, 0.6, alpha_value)
                
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.7,
                alphaMode='BLEND',
                baseColorFactor=base_color, roughnessFactor=0.7
            )

            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

            polygon_mesh.visual.face_colors = [70, 70, 70, 1.0]
            polygon_render = pyrender.Mesh.from_trimesh(polygon_mesh, smooth=False)
            sx, sy, tx, ty = [0.75, 0.75, 0, 0.10]

            camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))

            light = pyrender.DirectionalLight(color=[1,1,1], intensity=300)
            scene.add(mesh)

    c = np.pi / 2
    scene.add(polygon_render, pose=np.array([[1, 0, 0, 0],
                                             [0, np.cos(c), -np.sin(c), MINS[1].cpu().numpy()],
                                             [0, np.sin(c), np.cos(c), 0],
                                             [0, 0, 0, 1]]))

    light_pose = np.eye(4)
    light_pose[:3, 3] = [0, -1, 1]
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = [0, 1, 1]
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = [1, 1, 2]
    scene.add(light, pose=light_pose.copy())

    c = -np.pi / 6

    scene.add(camera, pose=[[1, 0, 0, (minx + maxx).cpu().numpy() / 2],
                            [0, np.cos(c), -np.sin(c), 3.0],
                            [0, np.sin(c), np.cos(c), 0.5 + max(4, minz.cpu().numpy() + (1.5 - MINS[1].cpu().numpy()) * 2, (maxx - minx).cpu().numpy())+1.0],
                            [0, 0, 0, 1]])

    ##활동량 적은거 단순 : 걷기 -> -0.5
    ## 태권도 활동량 많은거 -> 
    # render scene
    r = pyrender.OffscreenRenderer(960, 960)

    color, _ = r.render(scene, flags=RenderFlags.RGBA)
    Image.fromarray(color).save(os.path.join(outdir, f'{name}_render.png'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None, help='motion npy file dir')
    parser.add_argument('--motion-list', default=None, nargs="+", type=str, help="motion name list")
    args = parser.parse_args()

    if args.motion_list is None:
        filename_list = glob(os.path.join(args.dir, '*.npy'))
        filename_list = [os.path.splitext(os.path.split(fpath)[-1])[0] for fpath in filename_list]
    else:
        filename_list = args.motion_list
    for filename in filename_list:
        motions = np.load(os.path.join(args.dir, f'{filename}.npy'))
        if motions.shape == 4:
            motions = motions[0]

        # render_gif(motions, filename, outdir=args.dir, device_id=0)
        render_image(motions, filename, outdir=args.dir, device_id=0)
        