function [rgb,points3d,depthInpaint,imsize,seglabel]=read3dPointsSeg(data,anno2dSeg)
         depthVis = imread("."+data.depthpath);
         imsize = size(depthVis);
         depthInpaint = bitor(bitshift(depthVis,-3), bitshift(depthVis,16-3));
         depthInpaint = single(depthInpaint)/1000; 
         depthInpaint(depthInpaint >8)=8;
         [rgb,points3d,~,seglabel]=read_3d_pts_general_label(depthInpaint,data.K,size(depthInpaint),"."+data.rgbpath,anno2dSeg);
         points3d = (data.Rtilt*points3d')';
end