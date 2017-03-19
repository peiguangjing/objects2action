clear;
close all;

param.dataset = 'UCF-101';
param.datasetPath = 'datasets/UCF_101/';
param.videoResource = [param.datasetPath, 'dataset/'];
param.numberOfSplit = 3;
param.wordEmbeddingPath = 'semEmb/ucf_101/';
param.AWVObjectVectors = load([param.wordEmbeddingPath, 'AWV_Objects.mat']);
param.FWVObjectVectors = load([param.wordEmbeddingPath, 'FWV_Objects.mat']);
param.AWVActionVectors = load([param.wordEmbeddingPath, 'AWV_UCF101_Th14.mat']);
param.FWVActionVectors = load([param.wordEmbeddingPath, 'FWV_UCF101_Th14.mat']);
param.useFisherEncoding = true;
param.applySparsity = false;


if param.applySparsity == true
    param.videoSparsity = 10;
    param.actionSparsity = 100;
else 
    param.videoSparsity = 15293;
    param.actionSparsity = 15293;
end

averageAccuracy = 0;
for i = 1: param.numberOfSplit
    
    param.scores = load([param.datasetPath, 'scores/ObjRep_frmstep10_UCF101_test',num2str(i),'.mat']);
    param.datasetVideos = [param.datasetPath, 'splits/','testlist0',num2str(i),'.txt'];

    averageAccuracy = averageAccuracy + evalModel( param );
end

disp(['ZSL result for: ',param.dataset, ' dataset: ', num2str(averageAccuracy/param.numberOfSplit)]);



