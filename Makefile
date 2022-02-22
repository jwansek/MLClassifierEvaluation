all:
	javac -cp ".:../../../weka.jar" *.java

clean:
	rm -v *.class