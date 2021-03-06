function [] = sseBench(imgDir, gtDir, inDir, outDir)
% sseBench(imgDir, gtDir, inDir, outDir)
%
% Run Sum-Of-Squared Error benchmark on dataset.
%
% INPUT
%   imgDir  : folder containing original images
%   gtDir   : folder containing ground truth data.
%   inDir   : a collection of segmentations in a cell 'segs' stored in a mat file
%             - note that using an ultrametric contour map is not possible
%             with this evaluation function.
%   outDir  : folder where evaluation results will be stored
%
% David Stutz <david.stutz@rwth-aachen.de>

    iids = dir(fullfile(imgDir,'*.jpg'));
    for i = 1:numel(iids)
        evFile9 = fullfile(outDir, strcat(iids(i).name(1:end-4), '_ev9.txt'));
        if ~isempty(dir(evFile9))
            continue;
        end;

        inFile = fullfile(inDir, strcat(iids(i).name(1:end-4), '.mat'));
        gtFile = fullfile(gtDir, strcat(iids(i).name(1:end-4), '.mat'));
        imgFile = fullfile(imgDir, strcat(iids(i).name(1:end-4), '.jpg'));

        evaluation_sse_image(inFile, gtFile, imgFile, evFile9);
    end;

    collect_eval_sse(outDir);
    delete(sprintf('%s/*_ev9.txt', outDir));
end
