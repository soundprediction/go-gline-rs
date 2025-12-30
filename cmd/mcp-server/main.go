package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/soundprediction/go-gline-rs/pkg/gline"
	"github.com/spf13/cobra"
)

type ExtractEntitiesInput struct {
	Text   string   `json:"text"`
	Labels []string `json:"labels"`
}

var (
	modelID string
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "mcp-server",
		Short: "MCP Server for GLiNER",
		Long: `A Model Context Protocol (MCP) server that exposes the functionality 
of GLiNER models for entity extraction.`,
		Run: func(cmd *cobra.Command, args []string) {
			runServer(modelID)
		},
	}

	rootCmd.Flags().StringVar(&modelID, "model", "onnx-community/gliner_small-v2.1", "Hugging Face model ID")

	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func runServer(modelID string) {
	// Initialize gline
	if err := gline.Init(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to init gline: %v\n", err)
		os.Exit(1)
	}

	// Load model
	fmt.Fprintf(os.Stderr, "Loading model: %s...\n", modelID)
	model, err := gline.NewSpanModelFromHF(modelID)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load model %s: %v\n", modelID, err)
		os.Exit(1)
	}
	defer model.Close()
	fmt.Fprintf(os.Stderr, "Model loaded successfully.\n")

	// Create MCP server
	s := mcp.NewServer(&mcp.Implementation{
		Name:    "go-gline-rs-mcp",
		Version: "0.1.0",
	}, &mcp.ServerOptions{
		Capabilities: &mcp.ServerCapabilities{
			Tools: &mcp.ToolCapabilities{},
		},
	})

	// Add tool
	tool := &mcp.Tool{
		Name:        "extract_entities",
		Description: "Extract named entities from text using GLiNER model.",
	}

	mcp.AddTool(s, tool, func(ctx context.Context, req *mcp.CallToolRequest, input ExtractEntitiesInput) (*mcp.CallToolResult, any, error) {
		results, err := model.Predict([]string{input.Text}, input.Labels)
		if err != nil {
			return &mcp.CallToolResult{
				IsError: true,
				Content: []mcp.Content{
					&mcp.TextContent{
						Text: fmt.Sprintf("Inference error: %v", err),
					},
				},
			}, nil, nil
		}

		// Result for the single input text
		entities := results[0]

		jsonBytes, err := json.Marshal(entities)
		if err != nil {
			return &mcp.CallToolResult{
				IsError: true,
				Content: []mcp.Content{
					&mcp.TextContent{
						Text: fmt.Sprintf("Serialization error: %v", err),
					},
				},
			}, nil, nil
		}

		return &mcp.CallToolResult{
			Content: []mcp.Content{
				&mcp.TextContent{
					Text: string(jsonBytes),
				},
			},
		}, nil, nil
	})

	// Start server on stdio
	fmt.Fprintf(os.Stderr, "Starting MCP server on stdio...\n")
	if err := s.Run(context.Background(), &mcp.StdioTransport{}); err != nil {
		fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
		os.Exit(1)
	}
}
