from pathlib import Path
import pandas as pd


def prepare_data():
    # Use Path for robust cross-platform path handling
    base_path = Path("instacart")
    output_path = Path("clean_data.csv")

    if not base_path.exists():
        print(f"Error: Could not find directory at {base_path}")
        return

    print("--- Loading datasets ---")
    # Load metadata
    aisles = pd.read_csv(base_path / "aisles.csv")
    departments = pd.read_csv(base_path / "departments.csv")
    products = pd.read_csv(base_path / "products.csv")
    orders = pd.read_csv(base_path / "orders.csv")

    # Load order items (using 'train' set for a high-quality, manageable subset)
    # This includes the latest order for each user.
    order_items = pd.read_csv(base_path / "order_products__train.csv")

    print("--- Merging metadata (Products, Aisles, Departments) ---")
    # Enrich product information
    product_details = products.merge(aisles, on='aisle_id').merge(
        departments, on='department_id')

    print("--- Merging Order details ---")
    # Combine everything
    # 1. Start with items
    df = order_items.merge(product_details, on='product_id', how='left')

    # 2. Add order time/user info
    # Note: Instacart doesn't provide actual dates, so we use order_dow and hour_of_day
    df = df.merge(orders, on='order_id', how='left')

    # Clean up and select useful columns for correlation and seasonal analysis
    # We include aisle and department as they help find higher-level patterns
    df = df[[
        'user_id',
        'order_id',
        'product_name',
        'aisle',
        'department',
        'order_dow',              # Day of Week (for weekly trends)
        'order_hour_of_day',      # Hour (for daily trends)
        'days_since_prior_order',  # Recency (for churn/frequency patterns)
        'add_to_cart_order',      # Importance of product in that order
        'reordered'               # Loyalty indicator
    ]]

    # Rename columns for clarity if needed
    df.columns = [
        'user_id', 'order_id', 'product_name', 'aisle_name', 'department_name',
        'day_of_week', 'hour_of_day', 'days_since_last_order', 'cart_position', 'is_reorder'
    ]

    print(f"--- Final dataset shape: {df.shape} ---")
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Success! clean_data.csv is ready.")


if __name__ == "__main__":
    prepare_data()
