class PositionTracker:
    def __init__(self, initial_capital):
        """
        Initialize position tracker
        
        Args:
            initial_capital (float): Starting capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = None
        self.entry_price = 0
        self.position_size = 0
        self.trades = []
        
    def enter_position(self, price, time, position_type):
        """
        Enter new position
        
        Args:
            price (float): Entry price
            time (datetime): Entry time
            position_type (str): 'LONG' or 'SHORT'
        """
        # Implementation here
        pass
        
    def exit_position(self, price, time):
        """
        Exit current position
        
        Args:
            price (float): Exit price
            time (datetime): Exit time
        """
        # Implementation here
        pass
        
    def update_position(self, current_price):
        """
        Update position metrics
        
        Args:
            current_price (float): Current price
        """
        # Implementation here
        pass 