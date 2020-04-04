import torch
def create_nshot_task_label(k: int, q: int) -> torch.Tensor:
    """Creates an n-shot task label.

    Label has the structure:
        [0,1,2,..,k-1]*q

    # TODO: Test this

    # Arguments
        k: Number of classes in the n-shot classification task
        q: Number of query samples for each class in the n-shot classification task

    # Returns
        y: Label vector for n-shot task of shape [q * k, ]
    """
    y = torch.arange(k).repeat(q).long()
    return y