"""Summary
"""
import time
import logging
import types


def time_it(freq=1):
    """Summary

    Args:
        freq (TYPE): Description
    """
    class time_cls(object):

        """Summary

        Attributes:
            acc_time (float): Description
            call_count (int): Description
            class_name (str): Description
            freq (TYPE): Description
            func (TYPE): Description
        """

        def __init__(self, func):
            """Summary

            Args:
                func (TYPE): Description
            """
            # wraps(func)(self)
            self.__name__ = func.__name__
            self.__module__ = func.__module__
            self.__doc__ = func.__doc__
            self.func = func
            self.class_name = ""

            # used to calculate average run time
            self.call_count = 0
            self.acc_time = 0.0
            self.freq = freq

        def __call__(self, *args, **kwargs):
            """Summary

            Args:
                *args (TYPE): Description
                **kwargs (TYPE): Description

            Returns:
                TYPE: Description
            """
            start = time.time()
            # args contains the caller itself
            # print args[0] and you will see
            result = self.func(*args, **kwargs)
            end = time.time()

            self.call_count += 1
            self.acc_time += (end - start)
            if self.call_count > 0 and self.call_count % self.freq == 0:
                print("Avg time cost of %s%s is %f",
                              self.class_name, self.__name__,
                              self.acc_time / self.call_count)
                self.call_count = 0
                self.acc_time = 0.0

            return result

        def __get__(self, instance, cls):
            """Summary

            Args:
                instance (TYPE): Description
                cls (TYPE): Description

            Returns:
                TYPE: Description
            """
            if instance is None:
                return self
            else:
                self.class_name = "%s." % type(instance).__name__
                # bind this method to instance
                return types.MethodType(self, instance)

    return time_cls


@time_it(freq=2)
def test():
    """Summary
    """
    print("abc")


if __name__ == '__main__':
    test()
    test()
