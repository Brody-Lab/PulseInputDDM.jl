# Format for neural data

First, define a path to where the data you want to fit is located.

See [Loading data and fitting a choice model](@ref) for the expected format for .MAT files if one were fitting the choice model. In addition to those fields, for a neural model rawdata should also contain an extra field:

`rawdata.spike_times`: cell array containing the spike times of each neuron on an individual trial. The cell array will be length of the number of neurons recorded on that trial. Each entry of the cell array is a column vector containing the relative timing of spikes, in seconds. Zero seconds is the start of the click stimulus. Spikes before and after the click inputs should also be included.

The convention for fitting a model with neural model is that each session should have its own .MAT file. (This constrasts with the convention for the choice model, where a single .MAT file can contain data from different session). It's just easier this way, especially if different sessions have different number of cells.