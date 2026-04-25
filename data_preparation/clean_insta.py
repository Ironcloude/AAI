from pathlib import Path
import pandas as pd


def prepare_data():
    base_path = Path("instacart_src")
    output_path = Path("insta_clean_data.csv")

    if not base_path.exists():
        print(f"Error: Could not find directory at {base_path}")
        return

    print("--- Loading datasets ---")

    # Metadata
    aisles = pd.read_csv(base_path / "aisles.csv")
    departments = pd.read_csv(base_path / "departments.csv")
    products = pd.read_csv(base_path / "products.csv")
    orders = pd.read_csv(base_path / "orders.csv")

    # IMPORTANT: use BOTH prior + train
    prior = pd.read_csv(base_path / "order_products_prior.csv")
    train = pd.read_csv(base_path / "order_products_train.csv")

    # Label dataset origin
    prior["set"] = "prior"
    train["set"] = "train"

    # Combine
    order_items = pd.concat([prior, train], ignore_index=True)

    print("--- Merging product metadata ---")
    product_details = products.merge(aisles, on='aisle_id') \
                              .merge(departments, on='department_id')

    df = order_items.merge(product_details, on='product_id', how='left')

    print("--- Merging order info (CRITICAL) ---")
    df = df.merge(orders, on='order_id', how='left')

    # KEEP order_number (THIS FIXES EVERYTHING)
    df = df[[
        'user_id',
        'order_id',
        'order_number',              # sequence index
        'set',                       # prior/train flag
        'product_id',
        'product_name',
        'aisle',
        'department',
        'order_dow',
        'order_hour_of_day',
        'days_since_prior_order',
        'add_to_cart_order',
        'reordered'
    ]]

    df.columns = [
        'user_id',
        'order_id',
        'order_number',
        'eval_set',
        'product_id',
        'product_name',
        'aisle_name',
        'department_name',
        'day_of_week',
        'hour_of_day',
        'days_since_last_order',
        'cart_position',
        'is_reorder'
    ]

    # SORT = CRITICAL FOR SEQUENCES
    df = df.sort_values(by=['user_id', 'order_number', 'cart_position'])

    print(f"--- Final dataset shape: {df.shape} ---")
    print(f"Saving to {output_path}...")
    # df.to_csv(output_path, index=False)
    df.to_parquet("insta_clean_data.parquet", index=False)

    print("Sequence-ready dataset created!")


prepare_data()
