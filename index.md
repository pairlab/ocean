## OCEAN: Online Task Inference for Compositional Tasks with Context Adaptation

### Abstract
Real-world tasks often exhibit a compositional structure that contains a sequence of simpler sub-tasks. For instance, opening a door requires reaching, grasping, rotating, and pulling the door knob. Such compositional tasks require an agent to reason about the sub-task at hand while orchestrating global behavior accordingly. This can be cast as an online task inference problem, where the current task identity, represented by a context variable, is estimated from the agent's past experiences with probabilistic inference. Previous approaches have employed simple latent distributions, e.g., Gaussian, to model a single context for the entire task. However, this formulation lacks the expressiveness to capture the composition and transition of the sub-tasks. We propose a variational inference framework OCEAN to perform online task inference for compositional tasks. OCEAN models global and local context variables in a joint latent space, where the global variables represent a mixture of sub-tasks required for the task, while the local variables capture the transitions between the sub-tasks. Our framework supports flexible latent distributions based on prior knowledge of the task structure and can be trained in an unsupervised manner. Experimental results show that OCEAN provides more effective task inference with sequential context adaptation and thus leads to a performance boost on complex, multi-stage tasks


### Contributors and Contact

[Hongyu Ren](http://hyren.me/), [Yuke Zhu](https://www.cs.utexas.edu/~yukez/), [Jure Leskovec](https://cs.stanford.edu/people/jure/), [Anima Anandkumar](http://tensorlab.cms.caltech.edu/users/anima/), [Animesh Garg](http://animesh.garg.tech/). 
For questions, please email: [hyren@cs.stanford.edu](mailto:hyren@cs.stanford.edu) and [garg@cs.toronto.edu](mailto:garg@cs.toronto.edu)



