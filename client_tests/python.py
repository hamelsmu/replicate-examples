import replicate

output = replicate.run(
    "hamelsmu/honeycomb-2:453fb38babce2114a72c1eb8c03983610486b38c6872183687e43f18b432ffbb",
    input={
        "nlq": "EMISSING traces with the most spans",
        "cols": "['trace.span_id','span.num_links','span.num_events','trace.trace_id','span.kind','duration_ms','trace.parent_id','num_products','app.cart_total','telemetry.instrumentation_library','type','http.scheme','http.route','parent_name','error','net.transport','meta.signal_type','ip','k8s.pod.start_time','http.method','meta.annotation_type','http.target','http.request_content_length','http.host','http.flavor','k8s.pod.name','name','net.host.ip','service.name','k8s.pod.ip','k8s.pod.uid','library.name','net.host.port','library.version','rpc.service','http.user_agent','net.peer.ip','rpc.method','http.client_ip','message.type','net.host.name','app.user_id','k8s.namespace.name','http.server_name','message.id','k8s.node.name','net.peer.port','http.status_code','rpc.system','message.uncompressed_size']",
        "temperature": 1.0,
        "max_new_tokens": 5000
    }
)

if __name__ == "__main__":
    print(output)