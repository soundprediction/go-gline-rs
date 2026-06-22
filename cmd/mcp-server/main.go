package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/soundprediction/go-gline-rs/pkg/gliner2"
	"github.com/spf13/cobra"
)

// ExtractEntitiesInput is the input for the extract_entities tool.
type ExtractEntitiesInput struct {
	Text   string   `json:"text"`
	Labels []string `json:"labels"`
}

// ExtractRelationsInput is the input for the extract_relations tool.
type ExtractRelationsInput struct {
	Text          string   `json:"text"`
	RelationTypes []string `json:"relation_types"`
}

var (
	repo      string
	variant   string
	threshold float64
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "mcp-server",
		Short: "MCP server for GLiNER2 multi-task extraction",
		Long: `A Model Context Protocol (MCP) server exposing GLiNER2 information
extraction (named entities and relations) over ONNX Runtime via gliner2-rs.`,
		Run: func(cmd *cobra.Command, args []string) {
			runServer()
		},
	}

	rootCmd.Flags().StringVar(&repo, "repo", "SemplificaAI/gliner2-multi-v1-onnx", "Hugging Face model repo id")
	rootCmd.Flags().StringVar(&variant, "variant", "fp32_v2", "model variant subfolder (e.g. fp32_v2, fp16_v2)")
	rootCmd.Flags().Float64Var(&threshold, "threshold", 0.5, "confidence threshold")

	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func runServer() {
	fmt.Fprintf(os.Stderr, "Loading model %q (variant %q)…\n", repo, variant)
	eng, err := gliner2.NewFromHuggingFace(repo, variant)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load model %s: %v\n", repo, err)
		os.Exit(1)
	}
	defer eng.Close()
	fmt.Fprintf(os.Stderr, "Model loaded successfully.\n")

	s := mcp.NewServer(&mcp.Implementation{
		Name:    "go-gliner2-mcp",
		Version: "0.1.0",
	}, &mcp.ServerOptions{
		Capabilities: &mcp.ServerCapabilities{
			Tools: &mcp.ToolCapabilities{},
		},
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "extract_entities",
		Description: "Extract named entities of the given labels from text using GLiNER2.",
	}, func(ctx context.Context, req *mcp.CallToolRequest, in ExtractEntitiesInput) (*mcp.CallToolResult, any, error) {
		res, err := eng.Extract(in.Text, []gliner2.Task{gliner2.Entities(in.Labels...)}, float32(threshold), false)
		if err != nil {
			return errorResult(fmt.Sprintf("inference error: %v", err)), nil, nil
		}
		return jsonResult(res.Entities)
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "extract_relations",
		Description: "Extract relations of the given types between entities from text using GLiNER2.",
	}, func(ctx context.Context, req *mcp.CallToolRequest, in ExtractRelationsInput) (*mcp.CallToolResult, any, error) {
		tasks := make([]gliner2.Task, 0, len(in.RelationTypes))
		for _, rt := range in.RelationTypes {
			tasks = append(tasks, gliner2.Relations(rt, "head", "tail"))
		}
		res, err := eng.Extract(in.Text, tasks, float32(threshold), false)
		if err != nil {
			return errorResult(fmt.Sprintf("inference error: %v", err)), nil, nil
		}
		return jsonResult(res.Relations)
	})

	fmt.Fprintf(os.Stderr, "Starting MCP server on stdio…\n")
	if err := s.Run(context.Background(), &mcp.StdioTransport{}); err != nil {
		fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
		os.Exit(1)
	}
}

func jsonResult(v any) (*mcp.CallToolResult, any, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return errorResult(fmt.Sprintf("serialization error: %v", err)), nil, nil
	}
	return &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{Text: string(b)}},
	}, nil, nil
}

func errorResult(msg string) *mcp.CallToolResult {
	return &mcp.CallToolResult{
		IsError: true,
		Content: []mcp.Content{&mcp.TextContent{Text: msg}},
	}
}
