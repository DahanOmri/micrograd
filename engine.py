import math

class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data}), grad={self.grad}"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self): 
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other +(-self)
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'other type is not supported, it should be either int or float'
        out = Value(self.data ** other, (self, ), f'**{other}')

        def _backward():
            self.grad += (other * self.data) ** (other-1) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        out = Value(self.data * (other ** -1), (self, ), '/')
        return out

    def __rtruediv__(self, other):
        return other * self**-1
    
    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(max(0, self.data), (self, ), 'ReLU')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x)-1) / (math.exp(2 * x)+1)
        
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        def topo_sort(node):
            if node not in seen:
                seen.add(node)
                for neighbor in node._prev:
                    topo_sort(neighbor)
                sorted_nodes.append(node)

        seen = set()
        sorted_nodes = []
        topo_sort(self)
        self.grad = 1.0
        for node in reversed(sorted_nodes):
            node._backward()


if __name__ == "__main__":
    # Initialize a Value object with some data
    val1 = Value(10)
    val2 = Value(5)
    
    # Print the result and its details
    print(f"val1: {val1}")
    print(f"val2: {val2}")
    print(f"result (val1 + val2): {val1 + val2}")
    print(f"result (val1 - val2): {val1 - val2}")
    print(f"result (val1 * val2): {val1 * val2}")
    print(f"result (val1 ** 2): {val1 ** 2}")
    print(f"result (val1 / 2): {val1 / 2}")
    print(f"result (exp(val1)): {val1.exp()}")
    print(f"result (ReLU(val1)): {val1.relu()}")
    print(f"result (tanh(val1)): {val1.tanh()}")