function girths=computeGirth(points,faces,keyPointIndex)

girths = zeros(1,length(keyPointIndex));
for index = 1:length(faces(:,1))
    
    triangle(1,:) = points(faces(index,1),:);
    triangle(2,:) = points(faces(index,2),:);
    triangle(3,:) = points(faces(index,3),:);
    maxz = max(max(triangle(:,3)));
    minz = min(min(triangle(:,3)));
    maxx = max(max(triangle(:,1)));
    minx = min(min(triangle(:,1)));
    for kk=1:length(keyPointIndex)
        controlPoint = points(keyPointIndex(kk,1),:);
        %linelength = 0;
        
        switch kk
            case {1,4}
                leftPoint = points(keyPointIndex(kk,2),:);
                rightPoint = points(keyPointIndex(kk,3),:);
                if controlPoint(3) <= maxz & controlPoint(3) >= minz
                    if leftPoint(1)-10<=minx & rightPoint(1)+10 >=maxx
                        linelength = computeLengthOfTriangle(controlPoint(3),triangle(:,:));
                        girths(kk) = girths(kk) + linelength;
                    end
                end
                continue;
            case {2}
                leftPoint = points(keyPointIndex(kk,2),:);
                rightPoint = points(keyPointIndex(kk,3),:);
                if controlPoint(3) <= maxz & controlPoint(3) >= minz
                    if minx>=leftPoint(1)-5 & maxx<=rightPoint(1)+5
                        linelength = computeLengthOfTriangle(controlPoint(3),triangle(:,:));
                        girths(kk) = girths(kk) + linelength;
                        
                    end
                end
                continue;
            case {3,6}
                if controlPoint(3) <= maxz & controlPoint(3) >= minz
                    linelength = computeLengthOfTriangle(controlPoint(3),triangle(:,:));
                    girths(kk) = girths(kk) + linelength;
                end
                continue;
            case {5}
                leftPoint = points(keyPointIndex(kk,2),:);
                rightPoint = points(keyPointIndex(kk,3),:);
                if controlPoint(3) <= maxz & controlPoint(3) >= minz
                    if minx>=leftPoint(1)  & maxx<=rightPoint(1)
                        linelength = computeLengthOfTriangle(controlPoint(3),triangle(:,:));
                        girths(kk)= girths(kk) + linelength;
                    end
                end
                continue;
            case {7,8}
                
                leftPoint = points(keyPointIndex(kk,2),:);
                rightPoint = points(keyPointIndex(kk,3),:);
                
                
                if minx >= controlPoint(1)-5 & minz >= leftPoint(3)-20 & maxz <= rightPoint(3)+20
                    n1 = rightPoint-leftPoint;
                    n1 = n1/norm(n1);
                    n2 = n1;
                    n2(3) = 0;
                    n2 = n2/norm(n2);
                    
                    rotationAxis = cross(n1,n2);
                    angle = -acos(dot(n1,n2));
                    
                    u = rotationAxis/norm(rotationAxis);
                    sina = sin(angle);
                    cosa = cos(angle);
                    u1=u(1);
                    u2=u(2);
                    
                    rots=[cosa+u1*u1*(1-cosa)   u1*u2*(1-cosa)  u2*sina;
                        u1*u2*(1-cosa)      cosa+u2*u2*(1-cosa)     -u1*sina;
                        -u2*sina    u1*sina  cosa;
                        ];
                    angle = -angle;
                    sina = sin(angle);
                    cosa = cos(angle);
                    rotsv=[cosa+u1*u1*(1-cosa)   u1*u2*(1-cosa)  u2*sina;
                        u1*u2*(1-cosa)      cosa+u2*u2*(1-cosa)     -u1*sina;
                        -u2*sina    u1*sina  cosa;
                        ];
                    pp=leftPoint*rots;
                    triangle12 = triangle*rots;          
                    linelength = computeLengthOfTriangleWithRotation(pp(3),triangle12(:,:),rotsv);
                    girths(kk)= girths(kk) + linelength;
                end
                continue;
        end
    end
    
end %end index = 1:length(faces(:,1))
girths(1,3) = girths(1,3)/2.0;
girths




