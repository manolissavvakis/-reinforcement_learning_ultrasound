% Ultrasound confidence maps
% A. Karamalis, W. Wein, T. Klein, N. Navab: Ultrasound Confidence Maps
% using Random Walks, Medical Image Analysis, 16, 6, 1101 - 1112, 2012
% DOI: http://dx.doi.org/10.1016/j.media.2012.07.005
% 
% Chair for Computer Aided Medical Procedures (CAMP)
% Technische Universität München
% Written by: Athanasios Karamalis
% Email: karamali@in.tum.de
%
% THE WORK IS FOR RESEARCH AND NON-COMMERCIAL PURPOSES ONLY.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL CAMP BE LIABLE FOR ANY
% DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
%% Load example B-Mode NECK image 
load('data/neck.mat');

% Call confidence estimation for B-mode with default parameters
alpha = 2.0; beta = 90; gamma = 0.03;
[ map ] = confMap(img, alpha, beta, gamma);
%%
figure;
subplot(1,2,1); imagesc(img); colormap gray; axis off;
subplot(1,2,2); imagesc(map); colormap gray; axis off;

%% Load example B-Mode FEMUR image 
load('data/femur.mat');

alpha = 2.0; beta = 90; gamma = 0.06;
[ map ] = confMap(img, alpha, beta, gamma);

%%
%figure;
%subplot(1,2,1); imagesc(img); colormap gray; axis off;
%subplot(1,2,2); imagesc(map); colormap gray; axis off;

