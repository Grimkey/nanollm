package micrograd

import (
	"fmt"
	"math"
)

type Value struct {
	data     float64
	grad     float64 // Gradient of the value
	children []*Value
	op       string
	backward func() // Backpropagation function
}

func NewValue(data float64) *Value {
	return &Value{
		data:     data,
		grad:     0,
		children: nil,
		op:       "",
		backward: func() {},
	}
}

func convertToValue(data float64, children []*Value, op string) *Value {
	return &Value{
		data:     data,
		grad:     0,
		children: children,
		op:       op,
		backward: func() {},
	}
}

func (v *Value) Add(other *Value) *Value {
	out := convertToValue(v.data+other.data, []*Value{v, other}, "+")

	out.backward = func() {
		// Accumulate gradients proportionally
		v.grad += out.grad
		other.grad += out.grad
	}

	return out
}

func (v *Value) AddScalar(scalar float64) *Value {
	out := convertToValue(v.data+scalar, []*Value{v}, "+scalar")

	out.backward = func() {
		v.grad += out.grad
		// No need to accumulate gradients for the scalar as it's constant
	}

	return out
}

func (v *Value) Multiply(other *Value) *Value {
	out := convertToValue(v.data*other.data, []*Value{v, other}, "*")

	// Define the backpropagation logic for multiplication
	out.backward = func() {
		v.grad += other.data * out.grad
		other.grad += v.data * out.grad
	}

	return out
}

func (v *Value) MulScalar(scalar float64) *Value {
	return v.Multiply(convertToValue(scalar, nil, "scalar"))
}

func (v *Value) Pow(exp *Value) *Value {
	out := convertToValue(math.Pow(v.data, exp.data), []*Value{v, exp}, fmt.Sprintf("**%f", exp.data))

	out.backward = func() {
		// Gradient with respect to the base (v)
		if v.data != 0 {
			v.grad += exp.data * math.Pow(v.data, exp.data-1) * out.grad
		}

		// Gradient with respect to the exponent (exp)
		// Skip computing ln(v) if v <= 0 to avoid undefined behavior
		if v.data > 0 {
			exp.grad += math.Log(v.data) * out.data * out.grad
		} else {
			fmt.Printf("Skipping ln(v) for v=%f <= 0 in Pow backward\n", v.data)
		}
	}

	return out
}

func (v *Value) PowScalar(scalar float64) *Value {
	return v.Pow(convertToValue(scalar, nil, "scalar"))
}

func (v *Value) ReLU() *Value {
	out := convertToValue(0, []*Value{v}, "ReLU")
	if v.data > 0 {
		out.data = v.data
	}

	out.backward = func() {
		if v.data > 0 {
			v.grad += out.grad
		}
	}

	return out
}

func (v *Value) Backward() {
	// Initialize the gradient of the root node
	v.grad = 1.0

	// Topological sorting to find all dependencies in the graph
	var topo []*Value
	visited := make(map[*Value]bool)

	// Helper function for depth-first search
	var buildTopo func(*Value)
	buildTopo = func(node *Value) {
		if !visited[node] {
			visited[node] = true
			for _, child := range node.children {
				buildTopo(child)
			}
			topo = append(topo, node)
		}
	}

	// Build the topological order starting from this node
	buildTopo(v)

	fmt.Println("Topological Order:")
	for _, node := range topo {
		fmt.Printf("Node: %s, data=%f, grad=%f\n", node.op, node.data, node.grad)
	}

	// Flag to prevent redundant executions
	processed := make(map[*Value]bool)

	// Backward pass
	for i := len(topo) - 1; i >= 0; i-- {
		node := topo[i]
		if !processed[node] {
			processed[node] = true
			fmt.Printf("Processing Node: %s, data=%f, grad=%f\n", node.op, node.data, node.grad)
			node.backward()
		}
	}
}

func (v *Value) Neg() *Value {
	return v.Multiply(convertToValue(-1, nil, "neg"))
}

func (v *Value) Sub(other *Value) *Value {
	return v.Add(other.Neg())
}

func (v *Value) SubScalar(scalar float64) *Value {
	return v.Sub(convertToValue(scalar, nil, ""))
}

func (v *Value) Div(other *Value) *Value {
	return v.Multiply(other.PowScalar(-1))
}

func (v *Value) DivScalar(scalar float64) *Value {
	return v.Div(convertToValue(scalar, nil, "scalar"))
}

func (v *Value) String() string {
	return fmt.Sprintf("Value(data=%f, grad=%f, op='%s')", v.data, v.grad, v.op)
}
