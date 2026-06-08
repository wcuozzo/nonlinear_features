## TODO

Are the bias terms (where z(0) ends up) doing anything? Could they do anything even in theory?

Visualize features in 2d, compare to toy models of superposition plots
Could compare 2d visualizations when m=2 to UMAP/T-sne (send to Zack)

Sweep through other points, calculating metrics for each
First look at for points where m = 2 (or m = 1 for that matter), can visualize the results (to have images to associate with metrics)
Then also just have a table/phase diagram of metrics:
Linearity score: how much variance in z (latents) is explained by linear fit (how well can a linear function approximate mapping from x → z)
Nonlinear_gain: (mse_linear - mse_full)/(mse_linear). % of decrease in mse
Jacobian variance: the encoder behaves very differently in different regions of input space
Effective rank: of the dimensions available, how many are doing useful work?
Toy model phase diagrams (the n, m, l, sparsity?, corr?? sweep)
get a phase diagram (probably a series of 2d plots for each l, or maybe each n, m? unclear)
Produce some function, can itself be a neural net if necessary (but a functional form related high dimensional math/the exponential amount of space in higher dimensions would be better), that predicts the three metrics as a function of n, m, l. Then makes accurate predictions about the models in Csordas et al. and the llama models in the GLP paper.
These 3 papers give us some data points for free for where we should expecting linearity vs. nonlinearity in larger models. Do the functions we find apply?
Closed-form or semi-closed-form approximation of metric functions?

Think about feature sparsity/correlations. 
Reproduce toy model results for sparsity/corr (visualize in 2d)
How does it change phase diagrams? How does it change 2d feature visualizations?
Can we get to a 5 input function for different metrics? (n, m, l, sparsity, corr)
Closed-form or semi-closed-form approximation of metric functions?
How does this compare to the models in the 3 papers?

Look at gradient at origin (or z(0)). Features that have higher magnitudes need to escape quickly from low magnitude regions. Features with medium magnitudes need to escape low magnitude but then "slow down" before hitting higher magnitudes.
Shouldn't constraining to "write-linear" still allow for near perfect feature reconstruction (with magnitude superposition)? Compare practically
Even without magnitude superposition, couldn't you get pretty good feature reconstruction with densely packed circles (definitely with max sparsity, what about with medium sparsity?). Compare with model with normalized encoder.
Can any of these be broken down into “forces pulling toward linear representation” and “forces pulling toward nonlinear representation” terms?

Try signed features?

Could I train a diffusion/MFA model to learn better features for the toy model?
Could I visualize how these better model the features in 2d?

Metrics to distinguish 1d linear vs. still “mathematically linear” a la (Circuits Updates - July 2024) 
Compare with model that assumes mathematically linear but not 1d? (Claude suggests “higher-rank dictionary learning, or even careful PCA-based approaches”)

Beyond nonlinearity predictions, other connections with Csordás et al., diffusion (Guo), MLA results (shafran), and other literature?