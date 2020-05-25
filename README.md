# SocialNetworkInformationSpread

A model to simulate the spread of information through social networks, as described in Hu et al. (2017) and Young et al. (2020).

This repository contains the code for three infusion diffusion models:

- "Activation Jump Model" (AJM)
- "Independent Cascade Model" (ICM)
- 'Linear Threshold Model" (LTM)

For all models, the main inpute files are:

- fname = A .txt file of all social ties between pairs in the group (one row per pair)
- pl_fname = A .txt file of all the designate Peer Leaders in the group
- convert_fname = A .txt file of all the nodes that have been converted by an intervention
- nc_fname = A .txt file of all the nodes that have not been converted by an intervention
- all_nodes_fname = A .txt file of all the individuals in the entire group

Group all files in the same directory and run as follows:
- ICM and LTM can be run independently
- AJM must be run with a realistic 'h' value (which is determined by the 'Calculate_h' file

References:
Hu, L., Wilder, B., Yadav, A., Rice, E., & Tambe, M. (2017). Activating the" Breakfast Club": Modeling Influence Spread in Natural-World Social Networks. arXiv preprint arXiv:1710.00364. 

Young, L., Mayaud, J., Suen, S., Rice, E. (2020). Modeling the Dynamism of HIV Information Diffusion in Multiplex Networks of Homeless Youth. Social Networks.
