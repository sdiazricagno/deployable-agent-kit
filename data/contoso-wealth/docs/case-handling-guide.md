# Contoso Wealth case handling guide

## Teams
service-desk (first line), kyc-ops (KYC refreshes, onboarding documents, address changes), payments-ops (wires, transfers and holds), compliance (fraud, sanctions, suspicious activity).

## Severity
sev1: client funds at risk or suspected fraud. sev2: a held transfer past its review target, or a regulatory deadline. sev3: a standard client request. sev4: information only.

## Routing rules
Questions about held or delayed wires go to payments-ops. KYC refreshes and address changes go to kyc-ops. Suspected fraud goes to compliance at sev1.

## Closing cases
Resolve a case only once the client's request is complete, and add a comment with what was done.
