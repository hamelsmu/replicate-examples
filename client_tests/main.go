package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
)

func main() {
	apiURL := "https://api.replicate.com/v1/predictions"
	apiToken := os.Getenv("REPLICATE_API_TOKEN")

	// Struct for the JSON payload
	type Payload struct {
		Version      string `json:"version"`
		Input        Input  `json:"input"`
	}
	type Input struct {
		Nlq           string   `json:"nlq"`
		Cols          string   `json:"cols"`
		Temperature   float64  `json:"temperature"`
		MaxNewTokens  int      `json:"max_new_tokens"`
	}

	payload := Payload{
		Version: "453fb38babce2114a72c1eb8c03983610486b38c6872183687e43f18b432ffbb",
		Input: Input{
			Nlq: "traces with the most spans",
			Cols: "[\'trace.span_id\',\'span.num_links\',\'span.num_events\',\'trace.trace_id\',\'span.kind\',\'duration_ms\',\'trace.parent_id\',\'num_products\',\'app.cart_total\',\'telemetry.instrumentation_library\',\'type\',\'http.scheme\',\'http.route\',\'parent_name\',\'error\',\'net.transport\',\'meta.signal_type\',\'ip\',\'k8s.pod.start_time\',\'http.method\',\'meta.annotation_type\',\'http.target\',\'http.request_content_length\',\'http.host\',\'http.flavor\',\'k8s.pod.name\',\'name\',\'net.host.ip\',\'service.name\',\'k8s.pod.ip\',\'k8s.pod.uid\',\'library.name\',\'net.host.port\',\'library.version\',\'rpc.service\',\'http.user_agent\',\'net.peer.ip\',\'rpc.method\',\'http.client_ip\',\'message.type\',\'net.host.name\',\'app.user_id\',\'k8s.namespace.name\',\'http.server_name\',\'message.id\',\'k8s.node.name\',\'net.peer.port\',\'http.status_code\',\'rpc.system\',\'message.uncompressed_size\']",
			Temperature: 0.7,
			MaxNewTokens: 1000,
		},
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		fmt.Println("Error marshaling JSON:", err)
		return
	}

	req, err := http.NewRequest("POST", apiURL, bytes.NewBuffer(jsonData))
	if err != nil {
		fmt.Println("Error creating request:", err)
		return
	}

	req.Header.Set("Authorization", "Token "+apiToken)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Println("Error sending request to the server:", err)
		return
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		fmt.Println("Error reading response body:", err)
		return
	}

	fmt.Println("Response:", string(body))
}

