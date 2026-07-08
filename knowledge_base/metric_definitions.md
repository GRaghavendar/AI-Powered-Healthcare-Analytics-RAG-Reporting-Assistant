# Trauma Analytics Metric Definitions

## Purpose

This document defines the core measures used by the trauma analytics application. The measures are calculated from non-identifiable sample reporting data and aggregate outputs. They are intended for operational reporting, quality review, dashboard interpretation, and analyst question answering.

## Reporting Grain

Metrics may be summarized by:

- Reporting month
- Hospital or trauma center
- Region
- Adult or pediatric category
- Demographic segment
- Claim type or payer category

All dashboard views use aggregate calculations. The application does not expose names, addresses, medical record numbers, or other direct identifiers.

## Trauma Case Volume

Trauma case volume is the count of trauma registry visits for a selected reporting period, hospital, region, or segment.

Primary uses:

- Monitor reporting volume by month.
- Compare trauma center activity.
- Identify unusual submission patterns.
- Support executive-level workload reporting.

## Adult And Pediatric Distribution

Adult cases are visits where patient age is 18 or older. Pediatric cases are visits where patient age is below 18.

Primary uses:

- Separate adult and pediatric trauma workload.
- Review pediatric-capable facility activity.
- Support service planning and regional reporting.

## Fatality Rate

Fatality rate is calculated as:

`fatality cases / total trauma cases * 100`

Primary uses:

- Track aggregate mortality patterns.
- Compare hospital or regional trends.
- Support quality review discussions.

Important boundary:

This is an aggregate reporting metric. It is not a patient-level risk model and should not be used to predict an individual outcome.

## ICU Utilization Rate

ICU utilization rate is calculated as:

`ICU cases / total trauma cases * 100`

Primary uses:

- Summarize critical care utilization.
- Monitor changes across hospitals and months.
- Support resource planning conversations.

## Ventilator Utilization Rate

Ventilator utilization rate is calculated as:

`ventilator cases / total trauma cases * 100`

Related measure:

- Average ventilator days among cases with ventilator use.

Primary uses:

- Monitor respiratory support needs.
- Compare critical care intensity across reporting groups.

## Length Of Stay

Hospital length of stay is the number of inpatient days for a trauma visit. ICU length of stay is the number of ICU days.

Validation rule:

ICU days should not exceed total hospital days.

Primary uses:

- Track care intensity.
- Compare resource use by hospital or segment.
- Support operational performance review.

## ED And EMS Time Measures

Emergency department and EMS metrics include:

- ED minutes
- EMS response minutes
- EMS scene minutes
- EMS transport minutes
- Total pre-hospital minutes
- EMS recorded-value completeness

Primary uses:

- Review emergency response timing.
- Identify missing EMS documentation.
- Compare regional transport and ED patterns.

## Claims Utilization

Claims utilization summarizes:

- Claim count
- Allowed amount
- Paid amount
- Average paid amount
- Claim type
- Payer category

Primary uses:

- Understand cost and utilization patterns.
- Compare payment activity by service type.
- Support aggregate financial reporting.

Important boundary:

Claims metrics are for reporting and analysis only. They do not represent billing advice or payment adjudication.
