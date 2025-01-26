import cloudpickle

# import JobReturn
from hydra.core.utils import JobReturn



x = JobReturn()

p = cloudpickle.dumps(x)
print(p)


y = cloudpickle.loads(p)
print(y)