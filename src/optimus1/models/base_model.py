from abc import ABC, abstractmethod


class BasePlanningModel(ABC):
    """
    Base class for planning models.
    """

    @abstractmethod
    def planning(self):
        """
        Perform the planning process.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def replan(self):
        """
        Replan the model.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError


class BaseReflectionModel(ABC):
    """
    Base class for reflection models.
    """

    @abstractmethod
    def reflection(self):
        """
        Abstract method for performing reflection.
        Subclasses should implement this method.
        """
        raise NotImplementedError


class BaseActionModel(ABC):
    """
    Base class for action models.
    """

    @abstractmethod
    def action(self):
        """
        Perform an action.
        """
        raise NotImplementedError
