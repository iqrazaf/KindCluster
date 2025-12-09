Following is a cleaned, end-to-end, reproducible guide that combines kind cluster setup and k6 load test into one workflow.

It assumes a Ubuntu-like VM where you have sudo. 

Set up a local Kubernetes cluster with:

1: kind + 1 control-plane + 2 workers

2: Istio (demo profile)

3: Deploy Sock Shop, TeaStore, and Bookinfo apps

4: A single Istio Gateway exposing /sock, /tea, /book via http://localhost

5: A k6 spike-load test (s1_spike.js) that hits those three apps.