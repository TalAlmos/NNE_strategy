from typing import Dict, List, Optional
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Position:
    entry_time: datetime
    entry_price: float
    size: int
    direction: str  # 'LONG' or 'SHORT'
    stop_loss: float
    target: float
    initial_risk: float
    entry_type: str  # 'aggressive' or 'conservative'
    trend_data: Dict

class PositionTracker:
    def __init__(self, initial_capital: float = 100000.0):
        """Initialize position tracker
        
        Args:
            initial_capital: Starting capital amount
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: List[Position] = []
        self.closed_positions: List[Dict] = []
        self.daily_stats: Dict[str, Dict] = {}
        
    def add_position(self, 
                    entry_time: datetime,
                    entry_price: float,
                    size: int,
                    direction: str,
                    stop_loss: float,
                    target: float,
                    entry_type: str,
                    trend_data: Dict) -> None:
        """Add new position
        
        Args:
            entry_time: Entry timestamp
            entry_price: Entry price
            size: Position size
            direction: Trade direction
            stop_loss: Stop loss price
            target: Target price
            entry_type: Entry approach used
            trend_data: Associated trend data
        """
        initial_risk = abs(entry_price - stop_loss) * size
        
        position = Position(
            entry_time=entry_time,
            entry_price=entry_price,
            size=size,
            direction=direction,
            stop_loss=stop_loss,
            target=target,
            initial_risk=initial_risk,
            entry_type=entry_type,
            trend_data=trend_data
        )
        
        self.positions.append(position)
        
    def update_positions(self, current_bar: pd.Series) -> List[Dict]:
        """Update all positions with current price data
        
        Args:
            current_bar: Current price bar
            
        Returns:
            List of closed position details
        """
        closed = []
        
        for position in self.positions[:]:  # Copy list for safe removal
            result = self._check_position_exit(position, current_bar)
            if result:
                closed.append(result)
                self.positions.remove(position)
                self._update_capital(result['pnl'])
                
        return closed
    
    def update_stops(self, current_bar: pd.Series) -> None:
        """Update trailing stops for all positions
        
        Args:
            current_bar: Current price bar
        """
        for position in self.positions:
            if position.direction == 'LONG':
                new_stop = current_bar['High'] * 0.995  # 0.5% trailing stop
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
            else:
                new_stop = current_bar['Low'] * 1.005  # 0.5% trailing stop
                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop
    
    def process_eod(self, date: str) -> Dict:
        """Process end of day and generate statistics
        
        Args:
            date: Current date
            
        Returns:
            Daily statistics dictionary
        """
        # Calculate daily statistics
        daily_stats = {
            'date': date,
            'positions_opened': len(self.closed_positions),
            'positions_closed': len([p for p in self.closed_positions if p['exit_time'].date() == pd.Timestamp(date).date()]),
            'daily_pnl': sum(p['pnl'] for p in self.closed_positions if p['exit_time'].date() == pd.Timestamp(date).date()),
            'capital': self.current_capital
        }
        
        self.daily_stats[date] = daily_stats
        return daily_stats
    
    def get_position_summary(self) -> Dict:
        """Get summary of all positions
        
        Returns:
            Position summary dictionary
        """
        return {
            'open_positions': len(self.positions),
            'closed_positions': len(self.closed_positions),
            'total_pnl': sum(p['pnl'] for p in self.closed_positions),
            'win_rate': self._calculate_win_rate(),
            'avg_win': self._calculate_avg_win(),
            'avg_loss': self._calculate_avg_loss(),
            'risk_reward': self._calculate_risk_reward()
        }
    
    def _check_position_exit(self, position: Position, current_bar: pd.Series) -> Optional[Dict]:
        """Check if position should be exited
        
        Args:
            position: Position to check
            current_bar: Current price bar
            
        Returns:
            Exit details if position closed, None otherwise
        """
        exit_price = None
        exit_reason = None
        
        if position.direction == 'LONG':
            if current_bar['Low'] <= position.stop_loss:
                exit_price = position.stop_loss
                exit_reason = 'stop_loss'
            elif current_bar['High'] >= position.target:
                exit_price = position.target
                exit_reason = 'target'
        else:
            if current_bar['High'] >= position.stop_loss:
                exit_price = position.stop_loss
                exit_reason = 'stop_loss'
            elif current_bar['Low'] <= position.target:
                exit_price = position.target
                exit_reason = 'target'
                
        if exit_price:
            pnl = (exit_price - position.entry_price) * position.size
            if position.direction == 'SHORT':
                pnl = -pnl
                
            exit_details = {
                'entry_time': position.entry_time,
                'exit_time': current_bar.name,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'size': position.size,
                'direction': position.direction,
                'pnl': pnl,
                'exit_reason': exit_reason,
                'entry_type': position.entry_type,
                'initial_risk': position.initial_risk
            }
            
            self.closed_positions.append(exit_details)
            return exit_details
            
        return None
    
    def _update_capital(self, pnl: float) -> None:
        """Update current capital with trade P&L
        
        Args:
            pnl: Profit/loss amount
        """
        self.current_capital += pnl
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate percentage"""
        if not self.closed_positions:
            return 0.0
        winners = len([p for p in self.closed_positions if p['pnl'] > 0])
        return (winners / len(self.closed_positions)) * 100
    
    def _calculate_avg_win(self) -> float:
        """Calculate average winning trade"""
        winners = [p['pnl'] for p in self.closed_positions if p['pnl'] > 0]
        return sum(winners) / len(winners) if winners else 0.0
    
    def _calculate_avg_loss(self) -> float:
        """Calculate average losing trade"""
        losers = [p['pnl'] for p in self.closed_positions if p['pnl'] < 0]
        return sum(losers) / len(losers) if losers else 0.0
    
    def _calculate_risk_reward(self) -> float:
        """Calculate risk/reward ratio"""
        avg_win = self._calculate_avg_win()
        avg_loss = self._calculate_avg_loss()
        return abs(avg_win / avg_loss) if avg_loss != 0 else 0.0