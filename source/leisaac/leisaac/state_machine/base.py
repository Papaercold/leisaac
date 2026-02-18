"""Abstract base class for task state machines in LeIsaac."""

from abc import ABC, abstractmethod

import torch


class StateMachineBase(ABC):
    """Abstract base class for task state machines in LeIsaac.

    A state machine encapsulates the step-by-step control logic for a task.
    It is designed to be decoupled from the simulation control flow: the caller
    is responsible for calling :meth:`get_action`, stepping the environment, and
    then calling :meth:`advance` to progress the internal state.

    Typical usage::

        sm = MyStateMachine()
        env.reset()
        while not sm.is_episode_done:
            actions = sm.get_action(env)
            env.step(actions)
            sm.advance()
        sm.reset()
    """

    @abstractmethod
    def get_action(self, env) -> torch.Tensor:
        """Compute and return the action tensor for the current step.

        This method does **not** advance the internal state counter.
        Call :meth:`advance` after :meth:`env.step` to progress the machine.

        Args:
            env: The simulation environment instance. Must expose ``env.device``,
                ``env.num_envs``, and ``env.scene``.

        Returns:
            Action tensor of shape ``(num_envs, action_dim)``.
        """
        raise NotImplementedError

    @abstractmethod
    def advance(self) -> None:
        """Advance the internal step counter and manage state transitions.

        Should be called exactly once after each :meth:`env.step` call.
        Internally handles multi-phase and multi-object transitions.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset the state machine to its initial state.

        Should be called before starting a new episode.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_episode_done(self) -> bool:
        """Whether the state machine has completed a full episode cycle.

        Returns:
            ``True`` once the state machine has finished all phases of the task.
        """
        raise NotImplementedError
