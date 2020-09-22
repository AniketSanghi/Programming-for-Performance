import java.util.*;
import static java.util.stream.Collectors.*;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;

import org.antlr.v4.runtime.tree.ParseTreeProperty;
import org.antlr.v4.runtime.tree.TerminalNode;

// FIXME: You should limit your implementation to this class. You are free to add new auxilliary classes. You do not need to touch the LoopNext.g4 file.
class Analysis extends LoopNestBaseListener {

    // Possible types
    enum Types {
        Byte, Short, Int, Long, Char, Float, Double, Boolean, String
    }

    // Type of variable declaration
    enum VariableType {
        Primitive, Array, Literal
    }

    // Types of caches supported
    enum CacheTypes {
        DirectMapped, SetAssociative, FullyAssociative,
    }

    // auxilliary data-structure for converting strings
    // to types, ignoring strings because string is not a
    // valid type for loop bounds
    final Map<String, Types> stringToType = Collections.unmodifiableMap(new HashMap<String, Types>() {
        private static final long serialVersionUID = 1L;

        {
            put("byte", Types.Byte);
            put("short", Types.Short);
            put("int", Types.Int);
            put("long", Types.Long);
            put("char", Types.Char);
            put("float", Types.Float);
            put("double", Types.Double);
            put("boolean", Types.Boolean);
        }
    });

    // auxilliary data-structure for mapping types to their byte-size
    // size x means the actual size is 2^x bytes, again ignoring strings
    final Map<Types, Integer> typeToSize = Collections.unmodifiableMap(new HashMap<Types, Integer>() {
        private static final long serialVersionUID = 1L;

        {
            put(Types.Byte, 0);
            put(Types.Short, 1);
            put(Types.Int, 2);
            put(Types.Long, 3);
            put(Types.Char, 1);
            put(Types.Float, 2);
            put(Types.Double, 3);
            put(Types.Boolean, 0);
        }
    });

    // Map of cache type string to value of CacheTypes
    final Map<String, CacheTypes> stringToCacheType = Collections.unmodifiableMap(new HashMap<String, CacheTypes>() {
        private static final long serialVersionUID = 1L;

        {
            put("FullyAssociative", CacheTypes.FullyAssociative);
            put("SetAssociative", CacheTypes.SetAssociative);
            put("DirectMapped", CacheTypes.DirectMapped);
        }
    });

    public Analysis() {
    }

    private class SymbolTable {

        public class VariableData {
            String value = "";
            int size = -1;
            List<Long> dims = new ArrayList<Long>();
        }

        public class LoopData {
            long nestLevel = -1;
            long limit = -1;
            long stride = 0;
        }

        // Store all local variables with their current values
        HashMap<String, VariableData> data = new HashMap<String, VariableData>();
        HashMap<String, LoopData> loopData = new HashMap<String, LoopData>();

        // Current Being Declared Variable
        String currVal = "";
        int currForLoopNesting = -1;

        HashMap<String, List<String> > arrayAccess = new HashMap<String, List<String> >();
    }

    SymbolTable stable;
    List<HashMap<String, Long> > finalAnswer = new ArrayList<HashMap<String, Long> >();

    private class Solver {

        private long cacheSize, blockSize, numLines;
        String[] loopNestValues = new String[10];

        private long power(long a, long b) {
            long ans = 1;
            while(b != 0) {
                if(b%2 == 1) ans *= a;
                a = a*a;
                b /= 2;
            }
            return ans;
        }

        public Solver() {
            cacheSize = power(2, Long.parseLong(stable.data.get("cachePower").value));
            blockSize = power(2, Long.parseLong(stable.data.get("blockPower").value));

            if(stable.data.get("cacheType").value.charAt(1) == 'D') {
                numLines = 1;
            }
            else if(stable.data.get("cacheType").value.charAt(1) == 'S') {
                numLines = Long.parseLong(stable.data.get("setSize").value);
            } else {
                numLines = cacheSize / blockSize;
            }

            for(int i = 0; i<10; ++i) loopNestValues[i] = "";
            stable.loopData.forEach((key, value) -> {
                loopNestValues[(int)value.nestLevel] = key;
            });
        }

        private long numSets, sizeOfWord, numOfWordsInBlock, numOfWordsInLine;

        public long solve(String arrayName) {
            numSets = (cacheSize / blockSize) / numLines;
            sizeOfWord = power(2, stable.data.get(arrayName).size);
            
            if(blockSize >= sizeOfWord) numOfWordsInBlock = blockSize / sizeOfWord;
            else {
                long blocksToMerge = sizeOfWord/blockSize;
                numOfWordsInBlock = 1;

                while(blocksToMerge != 1) {
                    if(numSets > 1) numSets /= 2;
                    else numLines /= 2;

                    blocksToMerge /= 2;
                }
            }

            numOfWordsInLine = numOfWordsInBlock * numSets;

            return solveND(arrayName);
        }

        private long calculateIterationsProd(long a, long b) {
            long ans = 1;
            for(int i = (int)a; i <= b; ++i) {
                String it = loopNestValues[i];
                ans = ans * Math.max((stable.loopData.get(it).limit / stable.loopData.get(it).stride), 1);
            }
            return ans;
        }

        private long solveND(String arrayName) {
            List<Long> varsLoopLevels = new ArrayList<Long>();
            List<String> arrayAccessVars = stable.arrayAccess.get(arrayName);

            HashMap<String, Integer> varToDim = new HashMap<String, Integer>();

            for(int i = 1; i<arrayAccessVars.size(); ++i) {
                String it = arrayAccessVars.get(i);
                varsLoopLevels.add(stable.loopData.get(it).nestLevel);
                varToDim.put(it, i-1);
            }
            Collections.sort(varsLoopLevels);
            Collections.reverse(varsLoopLevels);

            long baseVar = stable.loopData.get(arrayAccessVars.get(arrayAccessVars.size() - 1)).nestLevel;

            boolean isCacheSufficient = true;
            long currNumOfElementsOnASet = 1;
            long ans = 1;

            List<Long> data = stable.data.get(arrayName).dims;

            for(int i = 0; i<varsLoopLevels.size(); ++i) {

                SymbolTable.LoopData itData = stable.loopData.get(loopNestValues[varsLoopLevels.get(i).intValue()]);

                int pos = varToDim.get(loopNestValues[varsLoopLevels.get(i).intValue()]);
                long actualStride = 1;
                if(pos == arrayAccessVars.size()-2) actualStride = 1;
                else {
                    while(pos < arrayAccessVars.size()-2) {
                        actualStride *= data.get(pos+1);
                        pos++;
                    }
                }

                long numOfLandsAtSetZero = Math.max(itData.limit*actualStride / (Math.max(numOfWordsInLine, itData.stride*actualStride)), 1);
                currNumOfElementsOnASet *= numOfLandsAtSetZero;

                ans *= Math.max((itData.limit / itData.stride), 1);

                if(isCacheSufficient) {
                    ans /= Math.max(Math.min(numOfWordsInBlock / (itData.stride * actualStride), itData.limit / itData.stride), 1);
                }

                isCacheSufficient = isCacheSufficient && (currNumOfElementsOnASet <= numLines);

                if(!isCacheSufficient) {
                    if(i+1 == varsLoopLevels.size()) {
                        ans *= calculateIterationsProd(0, varsLoopLevels.get(i)-1);
                    } else {
                        ans *= calculateIterationsProd(varsLoopLevels.get(i+1)+1, varsLoopLevels.get(i)-1);
                    }
                }
            }

            return ans;
        }
    }

    // FIXME: Feel free to override additional methods from
    // LoopNestBaseListener.java based on your needs.
    // Method entry callback
    @Override
    public void enterMethodDeclaration(LoopNestParser.MethodDeclarationContext ctx) {
        stable = new SymbolTable();
    }

    // End of testcase
    @Override
    public void exitMethodDeclaration(LoopNestParser.MethodDeclarationContext ctx) {

        Solver solver = new Solver();

        HashMap<String, Long> solution = new HashMap<String, Long>();

        stable.arrayAccess.forEach((key, value) -> {
            solution.put(key, solver.solve(key));
        });

        finalAnswer.add(solution);
    }

    @Override
    public void exitTests(LoopNestParser.TestsContext ctx) {

        // finalAnswer.forEach(solution -> {
        //     solution.forEach((key, value) -> {
        //         System.out.println(key + " -> " + value);
        //     });
        //     System.out.println("-------");
        // });

        try {
            FileOutputStream fos = new FileOutputStream("Results.obj");
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            // FIXME: Serialize your data to a file
            oos.writeObject(finalAnswer);
            oos.close();
        } catch (Exception e) {
            throw new RuntimeException(e.getMessage());
        }
    }

    @Override
    public void exitLocalVariableDeclaration(LoopNestParser.LocalVariableDeclarationContext ctx) {
    }

    @Override
    public void enterLocalVariableDeclaration(LoopNestParser.LocalVariableDeclarationContext ctx) {
        SymbolTable.VariableData data = stable.new VariableData();
        stable.data.put(ctx.getChild(1).getChild(0).getText(), data);
        stable.currVal = ctx.getChild(1).getChild(0).getText();
    }

    @Override
    public void exitVariableDeclarator(LoopNestParser.VariableDeclaratorContext ctx) {
        stable.data.get(stable.currVal).value = ctx.getChild(2).getText();
    }

    @Override
    public void exitArrayCreationExpression(LoopNestParser.ArrayCreationExpressionContext ctx) {
    }

    @Override
    public void exitDimExprs(LoopNestParser.DimExprsContext ctx) {
    }

    @Override
    public void exitDimExpr(LoopNestParser.DimExprContext ctx) {
        String val = ctx.getChild(1).getText();
        if(stable.data.get(val) != null) val = stable.data.get(val).value;
        stable.data.get(stable.currVal).dims.add(Long.parseLong(val));
    }

    @Override
    public void exitLiteral(LoopNestParser.LiteralContext ctx) {
    }

    @Override
    public void exitVariableDeclaratorId(LoopNestParser.VariableDeclaratorIdContext ctx) {
    }

    @Override
    public void exitUnannArrayType(LoopNestParser.UnannArrayTypeContext ctx) {
    }

    @Override
    public void enterDims(LoopNestParser.DimsContext ctx) {
    }

    @Override
    public void exitUnannPrimitiveType(LoopNestParser.UnannPrimitiveTypeContext ctx) {
        stable.data.get(stable.currVal).size = typeToSize.get(stringToType.get(ctx.getChild(0).getText()));
    }

    @Override
    public void exitNumericType(LoopNestParser.NumericTypeContext ctx) {
    }

    @Override
    public void exitIntegralType(LoopNestParser.IntegralTypeContext ctx) {
    }

    @Override
    public void exitFloatingPointType(LoopNestParser.FloatingPointTypeContext ctx) {
    }

    @Override
    public void exitForStatement(LoopNestParser.ForStatementContext ctx) {
        stable.currForLoopNesting -= 1;
    }

    @Override
    public void exitForInit(LoopNestParser.ForInitContext ctx) {
        stable.currForLoopNesting += 1;
        SymbolTable.LoopData data = stable.new LoopData();
        data.nestLevel = stable.currForLoopNesting;
        stable.loopData.put(ctx.getChild(0).getChild(1).getChild(0).getText(), data);
    }

    @Override
    public void exitForCondition(LoopNestParser.ForConditionContext ctx) {
        String limit = "0";
        if(stable.data.get(ctx.getChild(0).getChild(2).getText()) != null) {
            limit = stable.data.get(ctx.getChild(0).getChild(2).getText()).value;
        } else {
            limit = ctx.getChild(0).getChild(2).getText();
        }
        stable.loopData.get(ctx.getChild(0).getChild(0).getText()).limit = Long.parseLong(limit);
    }

    @Override
    public void exitRelationalExpression(LoopNestParser.RelationalExpressionContext ctx) {
    }

    @Override
    public void exitForUpdate(LoopNestParser.ForUpdateContext ctx) {
        String stride = "0";
        if(stable.data.get(ctx.getChild(0).getChild(2).getText()) != null) {
            stride = stable.data.get(ctx.getChild(0).getChild(2).getText()).value;
        } else {
            stride = ctx.getChild(0).getChild(2).getText();
        }
        stable.loopData.get(ctx.getChild(0).getChild(0).getText()).stride = Long.parseLong(stride);
    }

    @Override
    public void exitSimplifiedAssignment(LoopNestParser.SimplifiedAssignmentContext ctx) {
    }

    @Override
    public void exitArrayAccess(LoopNestParser.ArrayAccessContext ctx) {
        String arrayName = ctx.getChild(0).getText();
        stable.arrayAccess.put(arrayName, new ArrayList<String>());
        stable.arrayAccess.get(arrayName).add(0, Integer.toString(stable.currForLoopNesting));
        stable.arrayAccess.get(arrayName).add(1, ctx.getChild(2).getText());
        if(ctx.getChild(5) != null) stable.arrayAccess.get(arrayName).add(2, ctx.getChild(5).getText());
        if(ctx.getChild(8) != null) stable.arrayAccess.get(arrayName).add(3, ctx.getChild(8).getText());
    }

    @Override
    public void exitArrayAccess_lfno_primary(LoopNestParser.ArrayAccess_lfno_primaryContext ctx) {
        String arrayName = ctx.getChild(0).getText();
        stable.arrayAccess.put(arrayName, new ArrayList<String>());
        stable.arrayAccess.get(arrayName).add(0, Integer.toString(stable.currForLoopNesting));
        stable.arrayAccess.get(arrayName).add(1, ctx.getChild(2).getText());
        if(ctx.getChild(5) != null) stable.arrayAccess.get(arrayName).add(2, ctx.getChild(5).getText());
        if(ctx.getChild(8) != null) stable.arrayAccess.get(arrayName).add(3, ctx.getChild(8).getText());
    }
}

