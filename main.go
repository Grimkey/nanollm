package main

import (
	"bytes"
	"context"

	"github.com/goccy/go-graphviz"
)

func main() {
	ctx := context.Background()
	g, err := graphviz.New(ctx)
	if err != nil {
		panic(err)
	}

	graph, err := g.Graph()
	if err != nil {
		panic(err)
	}

	// create your graph

	// 1. write encoded PNG data to buffer
	var buf bytes.Buffer
	if err := g.Render(ctx, graph, graphviz.PNG, &buf); err != nil {
		panic(err)
	}

	// 2. get as image.Image instance
	_, err = g.RenderImage(ctx, graph)
	if err != nil {
		panic(err)
	}

	// 3. write to file directly
	if err := g.RenderFilename(ctx, graph, graphviz.PNG, "./graph.png"); err != nil {
		panic(err)
	}
}
