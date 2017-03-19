function p = videoRepresentation( objectScores, sparsity )
% Use semantic embeddings for zero-shot action classification in videos without using any video data or action annotations as prior knowledge.
% 
% INPUT
%
%   objectScores: object scores of related video which are obtained from AlexNet architecture
%   sparsity:     sparsification level of object scores. It might be
%                 helpful to ignore noisy data
%
% OUTPUT
%
%   p:            video representation for zero-shot action recognition
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

    [sortvals, sortidx] = sort(objectScores,'descend');

    B = zeros(size(objectScores), class(objectScores));

    for K = 1 : size(objectScores,2)
        B(sortidx(1:sparsity,K),K) = sortvals(1:sparsity,K);
    end

    p = B';

end
