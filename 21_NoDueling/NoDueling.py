from functools import partial
from anyrl.models import NatureDistQNetwork, noisy_net_dense

def no_dueling_models(session,
                      num_actions,
                      obs_vectorizer,
                      num_atoms=51,
                      min_val=-10,
                      max_val=10,
                      sigma0=0.5):
    """
    Create the models used for Rainbow without dueling.
    (https://arxiv.org/abs/1710.02298).
    Args:
      session: the TF session.
      num_actions: size of action space.
      obs_vectorizer: observation vectorizer.
      num_atoms: number of distribution atoms.
      min_val: minimum atom value.
      max_val: maximum atom value.
      sigma0: initial Noisy Net noise.
    Returns:
      A tuple (online, target).
    """
    def maker(name):
        return NatureDistQNetwork(session, num_actions, obs_vectorizer, name,
                                  num_atoms, min_val, max_val, dueling=False,
                                  dense=partial(noisy_net_dense, sigma0=sigma0))
    return maker('online'), maker('target')
