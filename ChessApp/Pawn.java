public class Pawn extends Piece{
    public Pawn(int x, int y, Side side, Board board) {
        // TODO: Call super constructor
        super(x,y,side,board);

    }

    @Override
    public boolean canMove(int destX, int destY) {

        //TODO: Check piecerules.md for the movement rule for this piece :)
        // need to implement that they can move 2 blocks on their first turn
        // need to also add condition for moving diagonally to capture a piece in other method
        if ((this.x == destX && this.y + 1 == destY) && (this.getSide() == Side.BLACK) && (board.get(x,y+1) == null)) {
            return true;
        }
        else if ((this.x == destX && this.y == 1 && this.y + 2 == destY) && (this.getSide() == Side.BLACK)) {
            return true;
        }
        else if ((this.x == destX && this.y == 6 && this.y - 2 == destY) && (this.getSide() == Side.WHITE)) {
            return true;
        }
        else if ((this.x == destX && this.y - 1 == destY) && (this.getSide() == Side.WHITE) && (board.get(x,y-1) == null)){
            return true;
        }
        //diagonal
        else if ((this.x + 1 == destX && this.y - 1 == destY) && (this.getSide() == Side.WHITE) && (board.get(x+1,y-1) != null)) {
            return true;
        }
        else if ((this.x - 1 == destX && this.y - 1 == destY) && (this.getSide() == Side.WHITE) && (board.get(x-1,y-1) != null)) {
            return true;
        }
        else if ((this.x + 1 == destX && this.y + 1 == destY) && (this.getSide() == Side.BLACK) && (board.get(x+1,y+1) != null)) {
            return true;
        }
        else if ((this.x - 1 == destX && this.y + 1 == destY) && (this.getSide() == Side.BLACK) && (board.get(x-1,y+1) != null)) {
            return true;
        }
        return false;
    }

    @Override
    public String getSymbol() {
        return this.getSide() == Side.BLACK ? "♟" : "♙";
    }
}
