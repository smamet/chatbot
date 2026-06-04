flowchart TD
    A([📧 Receive client email]) --> B{CLIENT EXIST ?}

    B -- YES --> C{NEW PRODUCTS ?}
    B -- NO --> D[WILL ASK CUSTOMER DETAILS IN EMAIL]

    D --> C
    C -- NO --> E[GENERATE QUOTE ERPNEXT]
    C -- YES --> F{HAVE INFO?}

    F -- YES --> G[Create quote]
    F -- NO --> H[Ask details]
    H --> I[SEND EMAIL]
    G --> I

    I --> A
