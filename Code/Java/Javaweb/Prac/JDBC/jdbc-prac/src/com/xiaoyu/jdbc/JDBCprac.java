package com.xiaoyu.jdbc;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;

public class JDBCprac {
    public static void main(String[] args) throws Exception {
        // 1. Load the MySQL JDBC driver
        Class.forName("com.mysql.cj.jdbc.Driver");

        // 2. Request for connection
        String url = "jdbc:mysql://localhost:3306/prac";
        String user = "root";
        String password = "MYSQLc-137040808";
        Connection conn = DriverManager.getConnection(url, user, password);

        // 3. Define SQL
        String sql = "update `student gpa` set gpa = 5.5 where id = 1";

        // 4. Create a statement
        Statement stmt = conn.createStatement();


        try {
            // 5. Start a transaction
            conn.setAutoCommit(false);

            // 6. Execute SQL
            int count = stmt.executeUpdate(sql); // rows affected

            // 7. Process the result
            System.out.println(count);

            // 8. Commit the transaction
            conn.commit();
        } catch (Exception e) {
            // 9. Rollback the transaction
            conn.rollback();
            throw new RuntimeException(e);
        }

        // 10. Close the resources
        stmt.close();
        conn.close();
    }
}
