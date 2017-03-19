function g = actionRepresentation( actionVectors, objectVectors, sparsity )

% Use semantic embeddings for zero-shot action classification in videos without using any video data or action annotations as prior knowledge.
% 
% INPUT
%   actionVectors: distributed word representations (e.g. word2vec) of related actions
%   objectScores:  object scores of related video which are obtained from AlexNet architecture
%   sparsity:      sparsification level of actions. This parameter might be helpful to ignore noisy data
%
% OUTPUT
%
%   g:              action representation for zero-shot action recognition
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

    g = objectVectors'*actionVectors;

    [sortvals, sortidx] = sort(g,'descend');

    B = zeros(size(g),class(g));

    for K = 1 : size(g,2)
        B(sortidx(1:sparsity,K),K) = sortvals(1:sparsity,K);
    end

    g = B';

end