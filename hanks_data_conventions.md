### Hanks data conventions from *Nature* paper

from `cell_packager.m`

```matlab
% get packaged data from cell. This includes event times and spike times
% without any further processing.
[array_data, vec_data, sessid, ratname] = package_pbups_phys(cellid,'save_to_file',0);
```

from `package_pbups_phys()`

```matlab
% INPUT:
%
%   cellid: id of cell
%
%
% OUTPUT:
% 
% array_data will contain a struct array with the following fields, where each
% element of the array represents an individual, non-violation trial:
%   spikes:         times of spikes relative to state_0
%	left_bups:      times of bups played on left relative to state_0
%   right_bups:     times of bups played on right relative to state_0
%   head_dtheta:    [n x 2] matrix with columns timestamps and angular head velocity
%   parsed_events:  parsed events for the trial
%
%
% vec_data will contain a structure with fields that are n trial vectors with 
% the following elements:
%   cpoke_start:    time relative to state_0 when center poke is initiated
%   cpoke_end:      time relative to state_0 when cpoke1 state ends
%   cpoke_out:      time relative to state_0 when rat leaves center poke
%   spoke_in:       time relative to state_0 when rat enters side poke
%   stim_start:     time relative to state_0 of first click on either side
%   gamma:          generative gamma
%   pokedR:         true for right poke choice
%   state_0_exits:  state_0 exit times based on state machine clock
%   state_0_entries:  state_0 entry times based on state machine clock
%   bup_diff:       right bups minus left bups
%   good:           index to good (included) trials (e.g., no cpoke vio)
%
% sessid:           sessid for this cellid
% ratname:          name of rat for this cellid
```

then, from `cell_packager.m`

```matlab
	data.trials.gamma(i)       = vec_data.gamma(i);
	data.trials.lpulses{i}     = array_data(i).left_bups(:)'  - cpoke_start(i);
	data.trials.rpulses{i}     = array_data(i).right_bups(:)' - cpoke_start(i);
    data.trials.bup_diff(i)    = vec_data.bup_diff(i);
	data.trials.spike_times{i} = array_data(i).spikes - cpoke_start(i);
	data.trials.correct_dir(i) = sign(data.trials.bup_diff(i));
	data.trials.rat_dir(i)     = 2*vec_data.pokedR(i) - 1; % +1 for R, -1 for L
	if data.trials.correct_dir(i) ~= 0,
		data.trials.hit(i)     = double(data.trials.correct_dir(i) == data.trials.rat_dir(i));
	else
		data.trials.hit(i)     = 0.5;   % assign 0.5 if there is no correct answer
	end;
	data.trials.cpoke_end(i)   = vec_data.cpoke_end(i)  - cpoke_start(i);
	data.trials.cpoke_out(i)   = vec_data.cpoke_out(i)  - cpoke_start(i);
	data.trials.stim_start(i)  = vec_data.stim_start(i) - cpoke_start(i);
	data.trials.rt(i)          = vec_data.spoke_in(i) - vec_data.cpoke_end(i);	
	data.trials.T(i)           = data.trials.cpoke_end(i) - data.trials.stim_start(i);
    data.trials.bup_diff_rate(i)  = data.trials.bup_diff(i) / data.trials.T(i);
```

After this, I do some post-processing, for example in `get_data_cells_ind_sess.m`

```matlab
            %bups, choice and trial time
            rawdata(k).leftbups = data(unique_id(j)).trials.lpulses{k} - ...
                data(unique_id(j)).trials.stim_start(k);
            rawdata(k).rightbups = data(unique_id(j)).trials.rpulses{k} - ...
                data(unique_id(j)).trials.stim_start(k);
            rawdata(k).T = data(unique_id(j)).trials.T(k);
            rawdata(k).pokedR = round(0.5 * (data(unique_id(j)).trials.rat_dir(k) + 1)); 
            rawdata(k).correct_dir = round(0.5 * (data(unique_id(j)).trials.correct_dir(k) + 1));          
            
            %loop over the cells from this session
            for l = 1:numel(which_cells)
                
                %if strcmp(data(which_cells(l)).region,'ppc')
                %    temp = data(which_cells(l)).trials.spike_times{k} - ...
                %        data(which_cells(l)).trials.stim_start(k) - 0.2;
                %elseif strcmp(data(which_cells(l)).region,'fof')
                %    temp = data(which_cells(l)).trials.spike_times{k} - ...
                %        data(which_cells(l)).trials.stim_start(k) - 0.1;
                %else
                    temp = data(which_cells(l)).trials.spike_times{k} - ...
                        data(which_cells(l)).trials.stim_start(k);
                %end
                
                temp2 = data(which_cells(l)).trials.T(k);
                
                %place data in the kth spot
                %reshpae is necessary to make sure that if all spike
                %were filtered out of the time window, then this entry
                %is 0x1
                %rawdata(k).St{l} = reshape(temp(temp >= 0 & temp <= temp2),[],1);
                %rawdata(k).St{l} = reshape(temp(temp >= -0.5 & temp <= temp2),[],1);
                rawdata(k).St{l} = reshape(temp,[],1);
                rawdata(k).cell{l} = cellids(which_cells(l));
                rawdata(k).sessid{l} = data(which_cells(l)).sessid;
                
            end
```

