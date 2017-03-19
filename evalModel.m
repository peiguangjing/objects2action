function [ accuracy ] = evalModel( param )
% Use semantic embeddings for zero-shot action classification in videos without using any video data or action annotations as prior knowledge.
% 
% INPUT
%
%   param:          related param inputs are explained in Manager.m script
%
% OUTPUT
%
%   accuracy:      zero-shot action classification accuracy of related dataset
%
% REFERENCES
% This is the "Objects2action: Classifying and localizing actions without any video example" implementation. 
% Related paper: (https://staff.fnwi.uva.nl/m.jain/pub/jain-objects2action-iccv2015.pdf)
% 
% NOTES
% * Authors provide word2vec models, embedded object-class and action class
% labels. To access them: https://staff.fnwi.uva.nl/m.jain/projects/Objects2action.html
%
% Implemented by Berkan Demirel, March 2017.


videoNames = readVideoNameFromFile( param.datasetVideos );

for i = 1: size(videoNames)
   
    C = strsplit(videoNames{i},'/');
    videoList{i} = C{1};
    
end

uniqueVideos = unique(videoList);

groundTruthLabels = getnameidx( uniqueVideos, videoList );

%videoParts = regexp(videoNames, '/', 'split');
%videoParts = arrayfun(@(n) videoParts{n}(1),1:size(videoNames,1));

objectScores = param.scores.vws;

if param.useFisherEncoding == false
    objectVectors = param.AWVObjectVectors.Obj_AWV;
    actionVectors = param.AWVActionVectors.Action_AWV;
else
    objectVectors = param.FWVObjectVectors.Obj_FWV;
    actionVectors = param.FWVActionVectors.Action_FWV;
end

%action representaton
g = actionRepresentation(actionVectors, objectVectors, param.actionSparsity);

%video representation
p = videoRepresentation(objectScores, param.videoSparsity);


objects2action = p*g';
[M,I] = max(objects2action,[],2);
uniqueActions = unique(groundTruthLabels);
averageResult = 0;


% This section calculates normalized top-1 per-class averaged accuracy, so
% it might be different from original paper.
for i = 1: size(uniqueActions,2)

    validVideos = find( groundTruthLabels == uniqueActions(i) );
    currResult = I(validVideos) == groundTruthLabels(validVideos)';
    averageResult = averageResult + sum(currResult)/size(currResult,1);
    
end

accuracy = averageResult/size(uniqueActions,2);
    

end

