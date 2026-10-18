# Private-only VPC: no internet gateway, no NAT. S3 is reached through a gateway endpoint (free).

locals {
  azs = slice(data.aws_availability_zones.available.names, 0, 2)
}

resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = "${var.name_prefix}-vpc" }
}

resource "aws_subnet" "private" {
  count                   = 2
  vpc_id                  = aws_vpc.this.id
  availability_zone       = local.azs[count.index]
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  map_public_ip_on_launch = false
  tags                    = { Name = "${var.name_prefix}-private-${local.azs[count.index]}" }
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.this.id
  tags   = { Name = "${var.name_prefix}-private" }
}

resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.this.id
  service_name      = "com.amazonaws.${var.region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.private.id]
  tags              = { Name = "${var.name_prefix}-s3" }
}

# Jobs get no ingress except from themselves (distributed jobs), and egress only to the S3 prefix
# list plus the VPC endpoints' own security group.
resource "aws_security_group" "jobs" {
  name        = "${var.name_prefix}-jobs"
  description = "SageMaker jobs: no internet, S3 via gateway endpoint only"
  vpc_id      = aws_vpc.this.id
  tags        = { Name = "${var.name_prefix}-jobs" }
}

resource "aws_vpc_security_group_ingress_rule" "self" {
  security_group_id            = aws_security_group.jobs.id
  referenced_security_group_id = aws_security_group.jobs.id
  ip_protocol                  = "-1"
}

resource "aws_vpc_security_group_egress_rule" "self" {
  security_group_id            = aws_security_group.jobs.id
  referenced_security_group_id = aws_security_group.jobs.id
  ip_protocol                  = "-1"
}

resource "aws_vpc_security_group_egress_rule" "s3" {
  security_group_id = aws_security_group.jobs.id
  prefix_list_id    = aws_vpc_endpoint.s3.prefix_list_id
  ip_protocol       = "-1"
}

# Optional interface endpoints (off by default; add only what a job proves it needs).
resource "aws_security_group" "endpoints" {
  count       = length(var.interface_endpoints) > 0 ? 1 : 0
  name        = "${var.name_prefix}-endpoints"
  description = "HTTPS from SageMaker jobs to interface endpoints"
  vpc_id      = aws_vpc.this.id
}

resource "aws_vpc_security_group_ingress_rule" "endpoints_https" {
  count                        = length(var.interface_endpoints) > 0 ? 1 : 0
  security_group_id            = aws_security_group.endpoints[0].id
  referenced_security_group_id = aws_security_group.jobs.id
  ip_protocol                  = "tcp"
  from_port                    = 443
  to_port                      = 443
}

resource "aws_vpc_security_group_egress_rule" "jobs_to_endpoints" {
  count                        = length(var.interface_endpoints) > 0 ? 1 : 0
  security_group_id            = aws_security_group.jobs.id
  referenced_security_group_id = aws_security_group.endpoints[0].id
  ip_protocol                  = "tcp"
  from_port                    = 443
  to_port                      = 443
}

resource "aws_vpc_endpoint" "interface" {
  for_each            = toset(var.interface_endpoints)
  vpc_id              = aws_vpc.this.id
  service_name        = "com.amazonaws.${var.region}.${each.key}"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = aws_subnet.private[*].id
  security_group_ids  = [aws_security_group.endpoints[0].id]
  private_dns_enabled = true
  tags                = { Name = "${var.name_prefix}-${each.key}" }
}
