import numpy as np
import os
import cv2
import open3d as o3d
import matplotlib as mpl
mpl.use("tkagg")


# rgb = cv2.imread('../data/Nikon/test1/DSC_1665.JPG')
rgb = cv2.imread('../data/TI/calibrate_data_11_11_2/01/LUCID_TRI032S-C_212401114__20211112182403477_image2417.jpg')
out = np.load("./test_out1.npy")
xyz = np.zeros((out.shape[0] * out.shape[1], 3), dtype=np.float32)
color = np.zeros((out.shape[0] * out.shape[1], 3), dtype=np.float32)

xyz[:, 0] = out[:, :, 0].flatten()
xyz[:, 1] = out[:, :, 1].flatten()
xyz[:, 2] = out[:, :, 2].flatten()
# xyz[:,0][np.where(xyz[:,2]<500)] = 0
# xyz[:,1][np.where(xyz[:,2]<500)] = 0
# xyz[:,2][np.where(xyz[:,2]<500)] = 0
xyz[:,0][np.where(xyz[:,2]>800)] = 0
xyz[:,1][np.where(xyz[:,2]>800)] = 0
xyz[:,2][np.where(xyz[:,2]>800)] = 0
color[:, 0] = rgb[:, :, 2].flatten()
color[:, 1] = rgb[:, :, 1].flatten()
color[:, 2] = rgb[:, :, 0].flatten()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(color/255.0)
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
# pcd = o3d.io.read_point_cloud('./test1_sync.ply')
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.run()
vis.destroy_window()

o3d.io.write_point_cloud("./test1_sync_1.ply", pcd)

