#%%writefile stock_prediction_transformer.py
# stock_prediction_transformer.py
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import json
import os
import gradio as gr
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

# Define constants
CONFIG_PATH = "config.json"
MODELS_DIR = "models"
DATA_DIR = "data"
DEFAULT_HISTORY_POINTS = 30  # Number of price points to use for prediction
DEFAULT_SEQ_LEN = 60  # Sequence length for transformer input
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")
# Initialize or load configuration
def init_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    else:
        default_config = {
            "stocks": {},
            "transformer_config": {
                "d_model": 128,
                "nhead": 8,
                "num_encoder_layers": 4,
                "num_decoder_layers": 4,
                "dim_feedforward": 256,
                "dropout": 0.2,
                "batch_size": 32,
                "learning_rate": 0.0001,
                "epochs": 100
            }
        }
        with open(CONFIG_PATH, 'w') as f:
            json.dump(default_config, f, indent=4)
        return default_config

# Save configuration
def save_config(config):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)

# Stock data handling class
class StockDataHandler:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fetch_data(self, period="max", interval="1d"):
        """Fetch stock data from Yahoo Finance and save to CSV"""
        try:
            print(f"Fetching data for {self.ticker}...")
            data = yf.download(self.ticker, period=period, interval=interval)
            if data.empty:
                print(f"No data available for {self.ticker}")
                return None

            print(f"Raw data shape: {data.shape}")
            print(f"Columns: {data.columns}")

            # Handle MultiIndex columns if they exist
            if isinstance(data.columns, pd.MultiIndex):
                # Select the Close price column for the ticker
                close_col = ('Close', self.ticker)
                if close_col in data.columns:
                    # Create a new DataFrame with just the Close prices
                    data = pd.DataFrame({
                        'Close': data[close_col]
                    }, index=data.index)
                else:
                    print(f"Could not find Close column for {self.ticker}")
                    return None

            try:
                # Convert to numeric using Series method
                data['Close'] = data['Close'].astype(float)
                data = data.dropna(subset=['Close'])  # Remove any rows with NaN values

                # Save only if we have valid data
                if not data.empty:
                    print(f"Downloaded {len(data)} data points for {self.ticker}")
                    # Save with datetime index in consistent format
                    data.index = pd.to_datetime(data.index).strftime('%Y-%m-%d')
                    data.to_csv(self.data_path)
                    print(f"Data for {self.ticker} saved to {self.data_path}")
                    return data
                return None

            except Exception as e:
                print(f"Error processing Close column: {str(e)}")
                print(f"Data columns after processing: {data.columns}")
                print(f"Sample of data:\n{data.head()}")
                return None

        except Exception as e:
            print(f"Error fetching data for {self.ticker}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

    def update_data(self):
        """Update data with latest prices"""
        if os.path.exists(self.data_path):
            existing_data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
            last_date = existing_data.index[-1]
            days_diff = (datetime.datetime.now() - last_date).days

            if days_diff > 0:
                # Fetch only new data
                new_data = yf.download(self.ticker, start=last_date + datetime.timedelta(days=1))
                if not new_data.empty:
                    updated_data = pd.concat([existing_data, new_data])
                    updated_data.to_csv(self.data_path)
                    return updated_data
                return existing_data
            return existing_data
        else:
            return self.fetch_data()

    def get_data(self):
        """Get data from file or fetch if not available"""
        try:
            if os.path.exists(self.data_path):
                print(f"Reading data from {self.data_path}")
                # Read CSV with explicit date parsing
                df = pd.read_csv(
                    self.data_path,
                    index_col=0,
                    parse_dates=True,
                    date_format='%Y-%m-%d'  # Specify exact date format
                )

                # Ensure Close column is numeric
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                df = df.dropna(subset=['Close'])  # Remove any rows with NaN values

                if df.empty:
                    print(f"No valid data found in file for {self.ticker}")
                    return self.fetch_data()

                print(f"Successfully loaded {len(df)} data points")
                return df
            else:
                return self.fetch_data()
        except Exception as e:
            print(f"Error reading data for {self.ticker}: {e}")
            # If there's an error reading the file, try fetching fresh data
            print("Attempting to fetch fresh data...")
            return self.fetch_data()

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators and add them to the DataFrame."""
        # Moving Averages
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_30'] = df['Close'].rolling(window=30).mean()

        # MACD
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Mid'] + (df['Close'].rolling(window=20).std() * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['Close'].rolling(window=20).std() * 2)

        return df

    def calculate_volatility(self, df):
        """Calculate historical volatility and add it to the DataFrame."""
        df['Volatility_10'] = df['Close'].pct_change().rolling(window=10).std() * np.sqrt(10)
        df['Volatility_30'] = df['Close'].pct_change().rolling(window=30).std() * np.sqrt(30)
        return df

    def prepare_data(self, seq_len=DEFAULT_SEQ_LEN, target_col='Close', train_split=0.8):
        """Prepare data for transformer model with additional features."""
        try:
            print(f"Preparing data for {self.ticker}...")
            df = self.get_data()
            if df is None or df.empty:
                print(f"No data available for {self.ticker}")
                return None, None, None

            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            df = self.calculate_volatility(df)

            # Calculate percentage changes instead of using absolute prices
            df['Returns'] = df[target_col].pct_change()
            df = df.dropna()

            print(f"Total data points available: {len(df)}")
            print(f"Required sequence length: {seq_len}")

            # Scale the returns data and additional features
            feature_columns = ['Returns', 'MA_10', 'MA_30', 'MACD', 'RSI', 'Volatility_10', 'Volatility_30']
            scaled_data = self.scaler.fit_transform(df[feature_columns])

            if len(scaled_data) <= seq_len:
                print(f"Not enough data points for {self.ticker}")
                return None, None, None

            # Create sequences
            x, y = [], []
            for i in range(len(scaled_data) - seq_len):
                x.append(scaled_data[i:i+seq_len])
                y.append(scaled_data[i+seq_len, 0])  # Assuming 'Returns' is the first column

            x, y = np.array(x), np.array(y)

            # Store the last price for prediction conversion
            self.last_price = float(df[target_col].iloc[-1])

            # Split into train and test sets
            train_size = int(len(x) * train_split)
            x_train, x_test = x[:train_size], x[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            return (x_train, y_train), (x_test, y_test), self.scaler
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            return None, None, None

# Custom Dataset for Transformer
class StockDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Transformer Model for Stock Price Prediction
class StockTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout, seq_len, input_dim):
        super(StockTransformer, self).__init__()

        self.d_model = d_model
        self.seq_len = seq_len

        # Update the input embedding to match the number of features
        self.embedding = nn.Linear(input_dim, d_model)  # Change 1 to input_dim

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        # Transformer layers
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                      dim_feedforward=dim_feedforward,
                                      dropout=dropout, batch_first=True),
            num_layers=num_encoder_layers
        )

        # Output layer
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size = x.size(0)

        # Embed the input sequence
        x = self.embedding(x)  # [batch_size, seq_len, d_model]

        # Add positional encoding
        x = x + self.pos_encoding

        # Pass through transformer encoder
        encoder_output = self.transformer_encoder(x)

        # Get the last time step output
        output = encoder_output[:, -1, :]

        # Pass through output layer
        return self.fc_out(output)

# Model Trainer
class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()

    def create_model(self, seq_len=DEFAULT_SEQ_LEN):
        """Create a new transformer model"""
        cfg = self.config["transformer_config"]
        input_dim = 7  # Update this to the number of features you are using
        self.model = StockTransformer(
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_encoder_layers=cfg["num_encoder_layers"],
            num_decoder_layers=cfg["num_decoder_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            seq_len=seq_len,
            input_dim=input_dim  # Pass the input dimension
        ).to(DEVICE)

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg["learning_rate"])
        return self.model

    def save_model(self, path):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path, seq_len=DEFAULT_SEQ_LEN):
        """Load model from disk"""
        if not os.path.exists(path):
            return self.create_model(seq_len)

        self.create_model(seq_len)
        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return self.model

    def train(self, dataloader, epochs=None):
        """Train the model"""
        if epochs is None:
            epochs = self.config["transformer_config"]["epochs"]

        self.model.train()
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

                # Reshape batch_y to match the output shape
                batch_y = batch_y.view(-1, 1)  # Reshape to [batch_size, 1]

                self.optimizer.zero_grad()

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        return losses

    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        predictions, actuals = [], []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

                outputs = self.model(batch_x)

                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'accuracy': 1 - (rmse / np.mean(actuals))  # A simple accuracy metric
        }

    def predict_next_n(self, input_seq, n_steps=1, scaler=None, last_price=None):
        """Predict next n price points"""
        self.model.eval()

        # Convert to tensor
        current_seq = torch.FloatTensor(input_seq).unsqueeze(0).to(DEVICE)

        predictions = []
        current_price = last_price

        with torch.no_grad():
            for _ in range(n_steps):
                # Get prediction (this is a return prediction)
                next_return = self.model(current_seq).cpu().numpy()[0]

                if scaler:
                    # Create a dummy array with the correct shape for inverse transformation
                    next_return_full = np.zeros((1, 7))  # 1 sample, 7 features
                    next_return_full[0, 0] = next_return.item()  # Use .item() to extract the scalar value

                    # Inverse transform the return using a DataFrame
                    next_return_df = pd.DataFrame(next_return_full, columns=['Returns', 'MA_10', 'MA_30', 'MACD', 'RSI', 'Volatility_10', 'Volatility_30'])
                    next_return = scaler.inverse_transform(next_return_df)[0][0]

                # Convert return to price
                next_price = current_price * (1 + next_return)
                predictions.append([next_price])
                current_price = next_price

                # Update sequence for next prediction
                # Normalize the return for the next prediction
                if scaler:
                    normalized_return_df = pd.DataFrame([[next_return] + [0] * 6], columns=['Returns', 'MA_10', 'MA_30', 'MACD', 'RSI', 'Volatility_10', 'Volatility_30'])
                    normalized_return = scaler.transform(normalized_return_df)[0]
                else:
                    normalized_return = next_return

                current_seq = torch.cat([
                    current_seq[:, 1:, :],
                    torch.FloatTensor(normalized_return).unsqueeze(0).unsqueeze(0).to(DEVICE)
                ], dim=1)

        return np.array(predictions)

# Stock Manager class
class StockManager:
    def __init__(self):
        self.config = init_config()
        self.trainer = ModelTrainer(self.config)

    def add_stock(self, ticker):
        """Add a new stock to the system"""
        if ticker in self.config["stocks"]:
            return f"Stock {ticker} already exists in the system."

        # Initialize stock handler and fetch data
        handler = StockDataHandler(ticker)
        data = handler.fetch_data()

        if data is None or data.empty:
            return f"Failed to fetch data for {ticker}."

        # Add stock to config
        self.config["stocks"][ticker] = {
            "added_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_trained": None,
            "accuracy": None,
            "model_path": os.path.join(MODELS_DIR, f"{ticker}_model.pt")
        }
        save_config(self.config)

        return f"Successfully added {ticker} to the system."

    def train_stock(self, ticker):
        """Train model for a specific stock"""
        if ticker not in self.config["stocks"]:
            return f"Stock {ticker} not found in the system."

        print(f"\nStarting training process for {ticker}")
        handler = StockDataHandler(ticker)

        # First check if we have data
        data = handler.get_data()
        if data is None or data.empty:
            return f"No data available for {ticker}. Please add the stock first."

        print(f"Data loaded for {ticker}. Preparing data for training...")
        data_result = handler.prepare_data()

        # Check if data preparation was successful
        if data_result[0] is None:
            return f"Failed to prepare data for {ticker}. Check the logs for details."

        (x_train, y_train), (x_test, y_test), scaler = data_result

        # Check if we have enough training data
        if len(x_train) < 10:  # Arbitrary small number as minimum
            return f"Not enough training data for {ticker}. Need more than 10 points, got {len(x_train)}."

        print(f"Creating datasets and dataloaders...")
        # Create datasets and dataloaders
        train_dataset = StockDataset(x_train, y_train)
        test_dataset = StockDataset(x_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["transformer_config"]["batch_size"],
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["transformer_config"]["batch_size"]
        )

        # Load or create model
        print(f"Loading/creating model...")
        model_path = self.config["stocks"][ticker]["model_path"]
        self.trainer.load_model(model_path)

        # Train model
        print(f"Starting training...")
        self.trainer.train(train_loader)

        # Evaluate model
        print(f"Evaluating model...")
        eval_results = self.trainer.evaluate(test_loader)

        # Save model
        print(f"Saving model...")
        self.trainer.save_model(model_path)

        # Update config
        self.config["stocks"][ticker]["last_trained"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.config["stocks"][ticker]["accuracy"] = eval_results["accuracy"]
        save_config(self.config)

        return f"Successfully trained model for {ticker}. Accuracy: {eval_results['accuracy']:.4f}"

    def evaluate_stock(self, ticker):
        """Evaluate model for a specific stock"""
        if ticker not in self.config["stocks"]:
            return f"Stock {ticker} not found in the system.", None

        handler = StockDataHandler(ticker)
        try:
            data_result = handler.prepare_data()
            if data_result[0] is None:
                return f"Failed to prepare data for {ticker}. Please check if the stock data is valid.", None

            (_, _), (x_test, y_test), scaler = data_result

            if x_test is None or len(x_test) == 0:
                return f"Insufficient data for evaluation of {ticker}.", None

            # Create dataset and dataloader
            test_dataset = StockDataset(x_test, y_test)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config["transformer_config"]["batch_size"]
            )

            # Load model
            model_path = self.config["stocks"][ticker]["model_path"]
            self.trainer.load_model(model_path)

            # Evaluate model
            eval_results = self.trainer.evaluate(test_loader)

            # Update config
            self.config["stocks"][ticker]["accuracy"] = eval_results["accuracy"]
            save_config(self.config)

            return f"Evaluation results for {ticker}:", eval_results
        except Exception as e:
            return f"Error evaluating {ticker}: {str(e)}", None

    def predict_stock(self, ticker, duration="1d"):
        """Predict stock price for a specific duration with improved volatility modeling"""
        if ticker not in self.config["stocks"]:
            return f"Stock {ticker} not found in the system.", None, None

        handler = StockDataHandler(ticker)
        data = handler.get_data()

        if data is None or data.empty:
            return f"No data available for {ticker}.", None, None

        # Prepare data and get the scaler
        data_result = handler.prepare_data()
        if data_result[0] is None:
            return f"Failed to prepare data for {ticker}.", None, None

        _, _, scaler = data_result

        # Get the last sequence of returns
        returns = data['Close'].pct_change().dropna()
        last_returns = returns[-DEFAULT_SEQ_LEN:].values.reshape(-1, 1)

        # Create a dummy array for the other features
        # Assuming the last returns should be combined with zeros for the other features
        last_returns_full = np.zeros((last_returns.shape[0], 7))  # 7 features
        last_returns_full[:, 0] = last_returns.flatten()  # Fill the first column with returns

        # Scale the last returns with the scaler
        last_returns_df = pd.DataFrame(last_returns_full, columns=['Returns', 'MA_10', 'MA_30', 'MACD', 'RSI', 'Volatility_10', 'Volatility_30'])
        scaled_returns = scaler.transform(last_returns_df)

        # Calculate historical volatility
        historical_volatility = returns.std() * np.sqrt(252)  # Annualized volatility

        # Get current price
        current_price = float(data['Close'].iloc[-1])

        # Process duration to number of steps
        duration_map = {
            "1d": 1, "3d": 3, "1w": 5, "1mo": 22,
            "3mo": 66, "6mo": 132, "1y": 252
        }
        n_steps = duration_map.get(duration, 1)

        # Load model and make base prediction
        model_path = self.config["stocks"][ticker]["model_path"]
        self.trainer.load_model(model_path)
        base_predictions = self.trainer.predict_next_n(scaled_returns, n_steps, scaler, current_price)

        # Apply more realistic modeling for longer term predictions
        predictions = []

        # For short term predictions (<=5 days), use the model's direct output
        if n_steps <= 5:
            predictions = base_predictions
        else:
            # For longer-term predictions, add realistic volatility and mean reversion
            predictions = np.zeros_like(base_predictions)
            predictions[0] = base_predictions[0]  # Keep first prediction as is

            # Calculate mean historical return (daily)
            mean_daily_return = returns.mean()

            # Get the trend direction from short-term predictions
            initial_trend = (base_predictions[0][0] / current_price) - 1

            # Random seed based on ticker for reproducibility
            np.random.seed(hash(ticker) % 2**32)

            # Decay factor: how quickly model predictions lose influence
            # Longer durations have stronger decay
            max_decay = 0.95 if n_steps > 66 else 0.85
            decay_rate = np.linspace(0, max_decay, n_steps)

            for i in range(1, n_steps):
                # Calculate weights between model prediction and random walk
                model_weight = 1 - decay_rate[i]
                random_weight = decay_rate[i]

                # Get model's predicted return for this step
                model_return = (base_predictions[i][0] / predictions[i-1][0]) - 1

                # Generate random component with historical volatility
                # More days = more volatility (scale with square root of time)
                daily_vol = historical_volatility / np.sqrt(252)
                time_scaled_vol = daily_vol * np.sqrt(1)  # Daily volatility

                # Random return with mean reversion component
                random_component = np.random.normal(
                    mean_daily_return * 0.7,  # Drift slightly toward historical mean
                    time_scaled_vol * 1.2     # Slightly higher than historical vol
                )

                # Combine model prediction with random component
                combined_return = (model_return * model_weight) + (random_component * random_weight)

                # For longer time periods, add mean reversion toward historical trends
                if i > 22:  # After about a month
                    # Calculate how far we've moved from start
                    current_total_return = (predictions[i-1][0] / current_price) - 1
                    # Mean reversion factor - pulls back toward reasonable returns
                    reversion_strength = 0.03 * (i / n_steps)  # Increases with time
                    # If we've moved too far from reasonable returns, pull back
                    if abs(current_total_return) > historical_volatility * np.sqrt(i/252):
                        # Apply mean reversion
                        mean_reversion = -np.sign(current_total_return) * reversion_strength
                        combined_return += mean_reversion

                # Calculate next price
                next_price = predictions[i-1][0] * (1 + combined_return)
                predictions[i][0] = next_price

        # Ensure predictions stay within reasonable bounds
        for i in range(len(predictions)):
            days = i + 1
            # Scale maximum change with square root of time
            max_change = historical_volatility * np.sqrt(days/252) * 2  # 2x the expected volatility range
            min_price = current_price * (1 - max_change)
            max_price = current_price * (1 + max_change)
            predictions[i][0] = np.clip(predictions[i][0], min_price, max_price)

        accuracy = self.config["stocks"][ticker]["accuracy"]
        return f"Price prediction for {ticker} for {duration}:", predictions, {
            'current_price': current_price,
            'accuracy': accuracy,
            'volatility': historical_volatility
        }

    def update_all_stocks(self):
        """Update data and retrain models for all stocks"""
        results = []

        for ticker in self.config["stocks"]:
            # Update data
            handler = StockDataHandler(ticker)
            updated_data = handler.update_data()

            if updated_data is None or updated_data.empty:
                results.append(f"Failed to update data for {ticker}.")
                continue

            # Evaluate current model
            _, eval_results = self.evaluate_stock(ticker)

            # Check if retraining is needed
            if eval_results is None or eval_results["accuracy"] < 0.7:  # Arbitrary threshold
                # Retrain model
                train_result = self.train_stock(ticker)
                results.append(f"Retrained {ticker}: {train_result}")
            else:
                results.append(f"No retraining needed for {ticker}. Current accuracy: {eval_results['accuracy']:.4f}")

        return results

    def get_stock_summary(self, ticker):
        """Get summary information for a stock"""
        if ticker not in self.config["stocks"]:
            return f"Stock {ticker} not found in the system.", None

        handler = StockDataHandler(ticker)
        data = handler.get_data()

        if data is None or data.empty:
            return f"No data available for {ticker}.", None

        # Get stock information
        info = self.config["stocks"][ticker]

        try:
            # Ensure we're working with numeric values
            current_price = float(data['Close'].iloc[-1])
            previous_price = float(data['Close'].iloc[-2])
            daily_change = ((current_price - previous_price) / previous_price) * 100

            # Calculate other changes
            try:
                week_change = ((current_price - float(data['Close'].iloc[-6])) / float(data['Close'].iloc[-6])) * 100
            except:
                week_change = None

            try:
                month_change = ((current_price - float(data['Close'].iloc[-23])) / float(data['Close'].iloc[-23])) * 100
            except:
                month_change = None

            try:
                year_change = ((current_price - float(data['Close'].iloc[-253])) / float(data['Close'].iloc[-253])) * 100
            except:
                year_change = None

            # Create chart
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data.index[-90:], data['Close'].iloc[-90:])
            ax.set_title(f"{ticker} - Last 90 Days")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.grid(True)

            return f"Summary for {ticker}:", {
                'ticker': ticker,
                'current_price': current_price,
                'daily_change': daily_change,
                'week_change': week_change,
                'month_change': month_change,
                'year_change': year_change,
                'last_trained': info["last_trained"],
                'model_accuracy': info["accuracy"],
                'chart': fig
            }
        except Exception as e:
            return f"Error processing data for {ticker}: {str(e)}", None

    def get_model_internals(self, ticker):
        """Get model internals for debugging"""
        if ticker not in self.config["stocks"]:
            return f"Stock {ticker} not found in the system.", None

        # Load model
        model_path = self.config["stocks"][ticker]["model_path"]
        model = self.trainer.load_model(model_path)

        # Get model architecture
        model_summary = str(model)

        # Get model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Get positional encoding visualization
        pos_encoding = model.pos_encoding.cpu().detach().numpy()

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(pos_encoding[0], cmap='viridis')
        ax.set_title("Positional Encoding Visualization")
        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel("Position")
        plt.colorbar(im)

        return f"Model internals for {ticker}:", {
            'model_summary': model_summary,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'pos_encoding_vis': fig,
            'config': self.config["transformer_config"]
        }

# Gradio UI implementation
def create_gradio_interface():
    stock_manager = StockManager()

    # User interface components

    def user_get_stock_info(ticker):
        message, info = stock_manager.get_stock_summary(ticker)
        if info is None:
            return message, None, None, None, None, None, None, None

        return (
            f"Current Price: ${info['current_price']:.2f}",
            f"Daily Change: {info['daily_change']:.2f}%",
            f"Weekly Change: {info['week_change']:.2f}%" if info['week_change'] is not None else "N/A",
            f"Monthly Change: {info['month_change']:.2f}%" if info['month_change'] is not None else "N/A",
            f"Yearly Change: {info['year_change']:.2f}%" if info['year_change'] is not None else "N/A",
            f"Model Accuracy: {info['model_accuracy']*100:.2f}%" if info['model_accuracy'] is not None else "N/A",
            info['chart'],
            f"Last trained: {info['last_trained']}" if info['last_trained'] is not None else "Not trained yet"
        )

    def user_predict_stock(ticker, duration):
        if not ticker:  # Check for empty input
            return "Please enter a valid ticker symbol.", None, None

        message, predictions, info = stock_manager.predict_stock(ticker, duration)
        if predictions is None:
            return message, None, None

        # Create prediction chart
        fig, ax = plt.subplots(figsize=(10, 5))

        # Calculate confidence intervals (e.g., +/- 1 and 2 standard deviations)
        days = np.arange(len(predictions))
        volatility = info.get('volatility', 0.3)  # Default if not available

        # Time-scaled volatility for confidence intervals
        current_price = float(info['current_price'])  # Ensure current_price is a float
        confidence_intervals = [np.sqrt(d/252) * volatility * current_price for d in days]

        # Plot confidence bands (68% and 95%)
        ax.fill_between(days,
                        [predictions[i][0] - 2*confidence_intervals[i] for i in range(len(predictions))],
                        [predictions[i][0] + 2*confidence_intervals[i] for i in range(len(predictions))],
                        color='lightblue', alpha=0.3, label='95% Confidence')

        ax.fill_between(days,
                        [predictions[i][0] - confidence_intervals[i] for i in range(len(predictions))],
                        [predictions[i][0] + confidence_intervals[i] for i in range(len(predictions))],
                        color='blue', alpha=0.3, label='68% Confidence')

        # Plot the mean prediction
        ax.plot(days, predictions, marker='o', color='darkblue', label='Predicted Price')

        ax.set_title(f"{ticker} - Price Prediction for {duration}")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Predicted Price")
        ax.grid(True)
        ax.legend()

        predicted_price = predictions[-1][0]
        change = ((predicted_price / current_price) - 1) * 100

        return (
            f"Predicted Price: ${predicted_price:.2f} ({change:.2f}%)",
            f"Model Accuracy: {info['accuracy']*100:.2f}%" if info['accuracy'] is not None else "N/A",
            fig
        )

    # Admin interface components

    def admin_add_stock(ticker):
        if not ticker or not isinstance(ticker, str):
            return "Please enter a valid ticker symbol (e.g., AAPL)."
        return stock_manager.add_stock(ticker.strip())

    def admin_train_stock(ticker):
        if not ticker:
            return "Please enter a valid ticker symbol."
        try:
            return stock_manager.train_stock(ticker.strip())
        except Exception as e:
            return f"Error training model for {ticker}: {str(e)}"

    def admin_evaluate_stock(ticker):
        message, results = stock_manager.evaluate_stock(ticker)
        if results is None:
            return message, None, None, None

        return (
            message,
            f"MSE: {results['mse']:.6f}",
            f"RMSE: {results['rmse']:.6f}",
            f"Accuracy: {results['accuracy']*100:.2f}%"
        )

    def admin_get_stocks():
        stocks = list(stock_manager.config["stocks"].keys())
        stock_info = []

        for ticker in stocks:
            info = stock_manager.config["stocks"][ticker]
            accuracy_str = f"{info['accuracy']*100:.2f}%" if info['accuracy'] is not None else "N/A"
            stock_info.append(f"{ticker}: Last trained: {info['last_trained'] or 'Never'}, Accuracy: {accuracy_str}")

        return "\n".join(stock_info) if stock_info else "No stocks found in the system."

    def admin_update_all_stocks():
        results = stock_manager.update_all_stocks()
        return "\n".join(results)

    def admin_get_model_internals(ticker):
        message, internals = stock_manager.get_model_internals(ticker)
        if internals is None:
            return message, None, None, None

        config_str = "\n".join([f"{k}: {v}" for k, v in internals['config'].items()])

        return (
            message,
            internals['model_summary'],
            f"Total parameters: {internals['total_params']}\nTrainable parameters: {internals['trainable_params']}",
            internals['pos_encoding_vis']
        )

    # Create user interface
    with gr.Blocks() as app:
        gr.Markdown("# Stock Price Prediction System")

        with gr.Tabs():
            with gr.Tab("User"):
                gr.Markdown("## Stock Information and Prediction")

                with gr.Row():
                    user_ticker_input = gr.Textbox(label="Stock Ticker (e.g., AAPL)")
                    user_info_btn = gr.Button("Get Stock Info")

                with gr.Row():
                    current_price = gr.Textbox(label="Current Price")
                    daily_change = gr.Textbox(label="Daily Change")
                    weekly_change = gr.Textbox(label="Weekly Change")
                    monthly_change = gr.Textbox(label="Monthly Change")

                with gr.Row():
                    yearly_change = gr.Textbox(label="Yearly Change")
                    model_accuracy = gr.Textbox(label="Model Accuracy")
                    last_trained = gr.Textbox(label="Last Trained")

                stock_chart = gr.Plot(label="Stock Price Chart")

                gr.Markdown("## Stock Price Prediction")

                with gr.Row():
                    prediction_ticker = gr.Textbox(label="Stock Ticker (e.g., AAPL)")
                    prediction_duration = gr.Dropdown(
                        choices=["5m", "15m", "30m", "1h", "4h", "1d", "3d", "1w", "1mo", "3mo", "6mo", "1y"],
                        label="Prediction Duration",
                        value="1d"
                    )
                    predict_btn = gr.Button("Predict")

                with gr.Row():
                    predicted_price = gr.Textbox(label="Predicted Price")
                    prediction_accuracy = gr.Textbox(label="Prediction Accuracy")

                prediction_chart = gr.Plot(label="Price Prediction Chart")

            with gr.Tab("Admin"):
                gr.Markdown("## Stock Management")

                with gr.Row():
                    admin_ticker_input = gr.Textbox(label="Stock Ticker (e.g., AAPL)")
                    add_stock_btn = gr.Button("Add Stock")
                    train_stock_btn = gr.Button("Train Stock")
                    evaluate_stock_btn = gr.Button("Evaluate Stock")

                add_stock_output = gr.Textbox(label="Add Stock Result")

                with gr.Row():
                    evaluation_output = gr.Textbox(label="Evaluation Result")
                    mse_output = gr.Textbox(label="MSE")
                    rmse_output = gr.Textbox(label="RMSE")
                    accuracy_output = gr.Textbox(label="Accuracy")

                gr.Markdown("## System Management")

                with gr.Row():
                    list_stocks_btn = gr.Button("List All Stocks")
                    update_all_btn = gr.Button("Update All Stocks")

                stocks_list_output = gr.Textbox(label="Stocks List")
                update_all_output = gr.Textbox(label="Update Results")

                gr.Markdown("## Model Debugging")

                with gr.Row():
                    debug_ticker_input = gr.Textbox(label="Stock Ticker (e.g., AAPL)")
                    get_internals_btn = gr.Button("Get Model Internals")

                with gr.Row():
                    internals_output = gr.Textbox(label="Model Information", lines=5)
                    model_summary = gr.Textbox(label="Model Architecture", lines=10)
                    model_params = gr.Textbox(label="Model Parameters")

                pos_encoding_plot = gr.Plot(label="Positional Encoding Visualization")

        # User interface events
        user_info_btn.click(
            user_get_stock_info,
            inputs=[user_ticker_input],
            outputs=[current_price, daily_change, weekly_change, monthly_change,
                    yearly_change, model_accuracy, stock_chart, last_trained]
        )

        predict_btn.click(
            user_predict_stock,
            inputs=[prediction_ticker, prediction_duration],
            outputs=[predicted_price, prediction_accuracy, prediction_chart]
        )

        # Admin interface events
        add_stock_btn.click(
            admin_add_stock,
            inputs=[admin_ticker_input],
            outputs=[add_stock_output]
        )

        train_stock_btn.click(
            admin_train_stock,
            inputs=[admin_ticker_input],
            outputs=[add_stock_output]
        )

        evaluate_stock_btn.click(
            admin_evaluate_stock,
            inputs=[admin_ticker_input],
            outputs=[evaluation_output, mse_output, rmse_output, accuracy_output]
        )

        list_stocks_btn.click(
            admin_get_stocks,
            inputs=[],
            outputs=[stocks_list_output]
        )

        update_all_btn.click(
            admin_update_all_stocks,
            inputs=[],
            outputs=[update_all_output]
        )

        get_internals_btn.click(
            admin_get_model_internals,
            inputs=[debug_ticker_input],
            outputs=[internals_output, model_summary, model_params, pos_encoding_plot]
        )

    return app

def cleanup_system():
    """Clean up existing data and start fresh"""
    print("Cleaning up existing data...")
    if os.path.exists('data'):
        shutil.rmtree('data')
    if os.path.exists('models'):
        shutil.rmtree('models')
    if os.path.exists('config.json'):
        os.remove('config.json')

    # Create fresh directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    print("System cleanup completed.")

# Main function to run the system
def main():
    print("Initializing Stock Price Prediction System...")
    # Clean up existing data before starting
    cleanup_system()

    app = create_gradio_interface()
    app.launch(share=True)

if __name__ == "__main__":
    main()