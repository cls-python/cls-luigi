def calculate_fulfillment_rate(demand, orders):
    total_demand = sum(demand)
    total_fulfilled = sum(orders)  # Assuming orders represent fulfilled demand
    return total_fulfilled / total_demand if total_demand > 0 else 0

def calculate_inventory_levels(orders):
    # This is a placeholder; you may want to implement a more complex logic
    return sum(orders)  # Assuming total orders represent inventory levels

def calculate_service_level(demand, orders):
    total_demand = sum(demand)
    total_fulfilled = sum(orders)  # Assuming orders represent fulfilled demand
    return total_fulfilled / total_demand if total_demand > 0 else 0

def calculate_lead_time():
    # Placeholder for lead time calculation; implement as needed
    return 0  # Return the actual lead time based on your logic