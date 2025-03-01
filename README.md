single agent DQN algorithm made by me

'''mermaid
classDiagram
    class GymInterface {
        +reset()
        +step(action)
        -get_current_state()
        +render()
        +close()
        +evaluate_model(model, env, num_episodes)
        -cal_cost_avg()
        -Visualize_invens(inventory, demand_qty, Mat_order, all_rewards)
        -export_state()
    }

    class DQN {
        -forward(x)
    }

    class ReplayBuffer {
        +push(state, action, reward, next_state, done)
        +sample(batch_size)
        -__len__()
    }
    
    class DQNAgent {
        +select_action(state, epsilon)
        +update(batch_size)
        -buffer : ReplayBuffer
    }
    
    class Inventory {
        +update_demand_quantity(daily_events)
        +update_inven_level(quantity_of_change, inven_type, daily_events)
        -_update_report(quantity_of_change, inven_type)
    }
    
    class Supplier {
        +deliver_to_manufacturer(procurement, material_qty, material_inventory, daily_events, lead_time_dict)
    }

    class Procurement {
        +receive_materials(material_qty, material_inventory, daily_events)
        +order_material(supplier, inventory, daily_events, lead_time_dict)
    }
    
    class Production {
        +process_items(daily_events)
    }
    
    class Sales {
        -_deliver_to_cust(demand_size, product_inventory, daily_events)
        +receive_demands(demand_qty, product_inventory, daily_events)
    }

    class Customer {
        +order_product(sales, product_inventory, daily_events, scenario)
    }

    class Cost {
        +cal_cost(instance, cost_type)
        +update_cost_log(inventorylist)
        +clear_cost()
    }

    class CustomEvalCallback {
        -_on_step()
    }
        
    GymInterface --> DQNAgent
    DQNAgent --> ReplayBuffer
    GymInterface --> Inventory
    GymInterface --> Production
    GymInterface --> Supplier
    GymInterface --> Procurement
    GymInterface --> Sales
