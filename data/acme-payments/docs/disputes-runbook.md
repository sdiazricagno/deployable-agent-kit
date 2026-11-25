# Acme Payments disputes runbook

## Chargeback response window
The merchant has 7 calendar days from the chargeback notification to respond with evidence. After that the dispute is lost by default. Cardholders can raise a chargeback up to 120 days after the transaction.

## Evidence to collect
Ask the merchant for the receipt or invoice, proof of delivery or service, and any written communication with the cardholder. Never ask for or store full card numbers.

## Dispute routing
Chargeback tickets go to the disputes queue with priority p3. Disputes above 5,000 USD are raised to p2.
