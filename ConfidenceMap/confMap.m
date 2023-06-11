%  	Compute confidence map
%   Input:
%       data:   RF ultrasound data (one scanline per column)
%       mode:   'RF' or 'B' mode data
%       alpha, beta, gamma: See Medical Image Analysis reference    
%   Output:
%       map:    Confidence map for data
function confMap(f_path, f_name)

	input_file = fullfile(f_path, f_name);
	path_to_go = fullfile(f_path, "go_conf");
    path_to_die = fullfile(f_path, "die_conf");

	while true
	
		disp(strcat('Confidence Worker: waiting for the job...'));
        fclose(fopen(fullfile(f_path, "conf_started"), 'w'));
		
		while (~isfile(path_to_go)) && (~isfile(path_to_die))
			pause(1);
		end

        if isfile(path_to_die)
	
			break;

		elseif isfile(path_to_go)

			disp('Preparing confidence estimation...');
			delete(path_to_go);
			
			data = csvread(input_file);
			alpha=2.0; beta=90; gamma=0.03; mode='B';

			data = double(data);
			data = (data - min(data(:))) ./ ((max(data(:))-min(data(:)))+eps);

			% Seeds and labels (boundary conditions)
			seeds = [];
			labels = [];

			sc = 1:size(data,2); %All elements

			%SOURCE ELEMENTS - 1st matrix row
			sr_up = ones(1,length(sc));
			seed = sub2ind(size(data),sr_up,sc);
			seed = unique(seed);
			seeds = [seeds seed];

			% Label 1
			label = zeros(1,length(seed));
			label = label + 1;
			labels = [labels label];

			%SINK ELEMENTS - last image row
			sr_down = ones(1,length(sc));
			sr_down = sr_down * size(data,1);
			seed = sub2ind(size(data),sr_down,sc);
			seed = unique(seed);
			seeds = [seeds seed];

			%Label 2
			label = zeros(1,length(seed));
			label = label + 2;
			labels = [labels label];

			% Attenuation with Beer-Lambert
			W = attenuationWeighting(data,alpha);

			disp('Solving confidence estimation problem, please wait...');

			% Apply weighting directly to image
			% Same as applying it individually during the formation of the Laplacian
			data = data .* W;

			% Find condidence values
			map = confidenceEstimation( data, seeds, labels, beta, gamma);

			% Only keep probabilities for virtual source notes.
			map = map(:,:,1);

			csvwrite(fullfile(f_path, 'confidence_map.csv'), map)
			disp('Confidence Map saved.');
			fclose(fopen(fullfile(f_path, "ready_conf"), 'w'));
		end
	end
	disp(strcat("Killing confidence map worker."))
	
end

