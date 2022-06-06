addpath(genpath('.'))
load('./Metadata/SUNRGBDMeta.mat')
load('./Metadata/SUNRGBD2Dseg.mat')
%%
% %%
% imageId = 31;
% data = SUNRGBDMeta(imageId);
% anno2d = SUNRGBD2Dseg(imageId);
% % figure,
% % imagesc(anno2d.seglabel);
% %%
% [rgb,points3d,depthInpaint,imsize,seglabel]=read3dPointsSeg(data, anno2d.seglabel);
% %%
% ptCloudMatrix = cat(2,points3d,rgb,seglabel);
% ptCloud = pointCloud(ptCloudMatrix(:,1:3));
% ptCloud.Color = cast(255*ptCloudMatrix(:,4:6),'uint8');
% ptCloud.Intensity = ptCloudMatrix(:,7);
% pcwrite(ptCloud ,'point_cloud_test_seglabel.pcd','Encoding','ascii');

%%
% V1
% imageID based point cloud salvation
% this is not readable by other libraries
% dataSize = size(SUNRGBD2Dseg);
% for imageId=1:3 %dataSize(2)
%     disp(imageId);
%     data = SUNRGBDMeta(imageId);
%     anno2d = SUNRGBD2Dseg(imageId);
%     [rgb,points3d,depthInpaint,imsize,seglabel]=read3dPointsSeg(data, anno2d.seglabel);
%     ptCloudMatrix = cat(2,points3d,rgb,seglabel);
%     ptCloud = pointCloud(ptCloudMatrix(:,1:3));
%     ptCloud.Color = cast(255*ptCloudMatrix(:,4:6),'uint8');
%     ptCloud.Intensity = ptCloudMatrix(:,7);
%     pcwrite(ptCloud ,['pcd_data/sunrgbd_pcd_' num2str(imageId) '.pcd'],'Encoding','binary');
% %   writematrix(ptCloud,'point_cloud_test.txt','FileType','text');
% end
%%
% V2
% imageID based point cloud salvation

dataSize = size(SUNRGBD2Dseg);
for imageId=1:dataSize(2)
    disp(imageId);
    data = SUNRGBDMeta(imageId);
    anno2d = SUNRGBD2Dseg(imageId);
    [rgb,points3d,depthInpaint,imsize,seglabel]=read3dPointsSeg(data, anno2d.seglabel);
    ptCloud = cat(2, points3d, rgb, seglabel);

    ptCloudFile = ['pcd_data_testing_2/raw/sunrgbd_xyzrgbl_' num2str(imageId) '.txt'];
%     fid = fopen(ptCloudFile,'w');
%     ptCloudSize = size(ptCloud);
%     fprintf(fid, num2str(ptCloudSize(1)));
%     fclose(fid);

%     writematrix(ptCloud,ptCloudFile,'FileType','text');
    %pcd format
    % concat stuff
    ptCloudMatrixRGB = cat(2,points3d,rgb);
    ptCloudMatrixI = cat(2,points3d,seglabel);
    % create point cloud structures
    ptCloudRGB = pointCloud(ptCloudMatrixRGB(:,1:3));
    ptCloudI = pointCloud(ptCloudMatrixI(:,1:3));
    % add color and labels (intensity) to the point cloud structures
    ptCloudRGB.Color = cast(ptCloudMatrixRGB(:,4:6),'uint8');
    ptCloudI.Intensity = ptCloudMatrixI(:,4);
    % save point clouds to pcd files in order to be readable by other
    % programs
    pcwrite(ptCloudRGB ,['pcd_rgb_data/sunrgbd_pcd_' num2str(imageId) '.pcd'],'Encoding','binary');
    pcwrite(ptCloudI ,['pcd_label_data/sunrgbd_pcd_' num2str(imageId) '.pcd'],'Encoding','binary');
end
