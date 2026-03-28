output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.fraud_mlops.name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.fraud_mlops.endpoint
  sensitive   = true
}

output "mlflow_bucket" {
  description = "MLflow artifacts bucket"
  value       = google_storage_bucket.mlflow_artifacts.name
}

output "pipeline_bucket" {
  description = "Kubeflow pipeline artifacts bucket"
  value       = google_storage_bucket.pipeline_artifacts.name
}

output "kubectl_command" {
  description = "Command to configure kubectl"
  value       = "gcloud container clusters get-credentials ${var.cluster_name} --zone ${var.zone} --project ${var.project_id}"
}