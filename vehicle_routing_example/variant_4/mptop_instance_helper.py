from string import Template
from collections import namedtuple

Customer = namedtuple('Customer',
                      ['lat', 'lon', 'scoring_value', 'input_value', 'service_time', 'visits', 'obligatory', 'opening',
                       'closing', 'week_day_regularity', 'category_abc'])
Sales_Person = namedtuple('Sales_Person',
                          ['lat', 'lon', 'earliest', 'latest', 'time_limit', 'days_per_week', 'nb_weeks'])

body = Template("""{
  "nbWeeks": ${nb_weeks},
  "dimaPath": "${dima_result_name}",
  "daysPerWeek": ${days_per_week},
  "initialAssignment": "none",
  "vehicles": [
    {
      "extId": "${sales_person_id}",
      "posX": ${sales_person_lon},
      "posY": ${sales_person_lat},
      "earliest": ${sales_person_earliest},
      "latest": ${sales_person_latest},
      "timeLimit": ${sales_person_time_limit},
      "excludedDays": [
        
      ]
    }
  ],
  "nodes": [
  ${customer_nodes}
  ]
}""")

customer_node = Template("""  {
      "extId": "${customer_id}",
      "posX": ${customer_lon},
      "posY": ${customer_lat},
      "value": ${customer_scoring_value},
      "inputValue": ${customer_input_value},
      "serviceTime": ${customer_service_time},
      "visits": ${customer_visits},
      "obligatory": ${customer_obligatory},
      "weekPattern": [
        {
          "pattern": "1000"
        },
        {
          "pattern": "0100"
        },
        {
          "pattern": "0010"
        },
        {
          "pattern": "0001"
        }
      ],
      "weekdayPattern": [
        {
          "pattern": "10000"
        },
        {
          "pattern": "01000"
        },
        {
          "pattern": "00100"
        },
        {
          "pattern": "00010"
        },
        {
          "pattern": "00001"
        }
      ],
      "opening": ${customer_opening},
      "closing": ${customer_closing},
      "weekdayRegularity": "${customer_week_day_regularity}",
      "categoryABC": ${customer_category_abc}
  }""")
