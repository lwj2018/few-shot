Refer to https://github.com/huang-paper-code/fsl-global

### Todos
- Experiments on miniImagenet
- Experiments on omnilog
<!-- - Implemention of other methods (e.g. Matching Networks,Relation Networks) -->

### To test a new fsl method, you have to
- Implement the new model, also implement its get_optim_policies() method
- Implement train, val, test function for this new model
- Implement a new criterion if neccessary
- Implement new trainning, evaluation etc. scripts, and you should reconfig these items:
    - the model
    - train & eval functions
    - criterion
    - some hyper parameters if neccessary

