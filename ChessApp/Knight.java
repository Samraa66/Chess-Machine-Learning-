public class Knight extends Piece{
    public Knight(int x, int y, Side side, Board board) {
        // TODO: Call super constructor
        super(x,y,side,board);

    }

    @Override
    public boolean canMove(int destX, int destY) {

        //TODO: Check piecerules.md for the movement rule for this piece :)
        // need to implement that knights can hop over pieces
        if ((Math.abs(this.x - destX) == 2 && Math.abs(this.y  - destY) == 1) || (Math.abs(this.x - destX) == 1 && Math.abs(this.y  - destY) == 2)){
            return true;
        }
        return false;
    }

    @Override
    public String getSymbol() {
        return this.getSide() == Side.BLACK ? "♞" : "♘" ;
    }
}
